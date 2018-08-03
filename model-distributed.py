from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import sys
import tempfile
import time

import tensorflow as tf
import random
from tensorflow.contrib.framework import arg_scope
import numpy as np
from data import dataIterator, load_dict, prepare_data
import copy
import re
import os

rng = np.random.RandomState(int(time.time()))

def create_done_queue(ps_task_index, worker_count):

    with tf.device("/job:ps/task:%d/cpu:0" % (ps_task_index)):
        return tf.FIFOQueue(worker_count, tf.int32, shared_name="done_queue" + str(ps_task_index))

def create_done_queues(ps_count, worker_count):
    return [create_done_queue(ps_task_index, worker_count) for ps_task_index in range(ps_count)]

def cmp_result(label,rec):
    dist_mat = np.zeros((len(label)+1, len(rec)+1),dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i,j] = min(hit_score, ins_score, del_score)

    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)

def process(recfile, labelfile, resultfile):
    total_dist = 0
    total_label = 0
    total_line = 0
    total_line_rec = 0
    rec_mat = {}
    label_mat = {}
    with open(recfile) as f_rec:
        for line in f_rec:
            tmp = line.split()
            key = tmp[0]
            latex = tmp[1:]
            rec_mat[key] = latex
    with open(labelfile) as f_label:
        for line in f_label:
            tmp = line.split()
            key = tmp[0]
            latex = tmp[1:]
            label_mat[key] = latex
    for key_rec in rec_mat:
        label = label_mat[key_rec]
        rec = rec_mat[key_rec]
        dist, llen = cmp_result(label, rec)
        total_dist += dist
        total_label += llen
        total_line += 1
        if dist == 0:
            total_line_rec += 1
    wer = float(total_dist)/total_label
    sacc = float(total_line_rec)/total_line

    f_result = open(resultfile,'w')
    f_result.write('WER {}\n'.format(wer))
    f_result.write('ExpRate {}\n'.format(sacc))
    f_result.close()


def main(args):

    def norm_weight(fan_in, fan_out):
        W_bound = np.sqrt(6.0 / (fan_in + fan_out))
        return np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(fan_in, fan_out)), dtype=np.float32)
    
    
    def conv_norm_weight(nin, nout, kernel_size):
        filter_shape = (kernel_size[0], kernel_size[1], nin, nout)
        fan_in = kernel_size[0] * kernel_size[1] * nin
        fan_out = kernel_size[0] * kernel_size[1] * nout
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        W = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=np.float32)
        return W.astype('float32')
    
    
    def ortho_weight(ndim):
        W = np.random.randn(ndim, ndim)
        u, s, v = np.linalg.svd(W)
        return u.astype('float32')
    
    
    class Watcher_train():
        def __init__(self, blocks,             # number of dense blocks
                    level,                     # number of levels in each blocks
                    growth_rate,               # growth rate in DenseNet paper: k
                    training,
                    dropout_rate=0.2,          # keep-rate of dropout layer
                    dense_channels=0,          # filter numbers of transition layer's input
                    transition=0.5,            # rate of comprssion
                    input_conv_filters=48,     # filter numbers of conv2d before dense blocks
                    input_conv_stride=2,       # stride of conv2d before dense blocks
                    input_conv_kernel=[7,7]):  # kernel size of conv2d before dense blocks
            self.blocks = blocks
            self.growth_rate = growth_rate
            self.training = training
            self.dense_channels = dense_channels
            self.level = level
            self.dropout_rate = dropout_rate
            self.transition = transition
            self.input_conv_kernel = input_conv_kernel
            self.input_conv_stride = input_conv_stride
            self.input_conv_filters = input_conv_filters
    
        def bound(self, nin, nout, kernel):
            fin = nin * kernel[0] * kernel[1]
            fout = nout * kernel[0] * kernel[1]
            return np.sqrt(6. / (fin + fout))
    
    
        def dense_net(self, input_x, mask_x):
    
            #### before flowing into dense blocks ####
            x = input_x
            with tf.variable_scope('pre_conv') as scope:
                limit = self.bound(1, self.input_conv_filters, self.input_conv_kernel)
                x = tf.layers.conv2d(x, filters=self.input_conv_filters, strides=self.input_conv_stride,
                    kernel_size=self.input_conv_kernel, padding='SAME', data_format='channels_last', use_bias=False, kernel_initializer=tf.random_uniform_initializer(-limit, limit, dtype=tf.float32))
                mask_x = mask_x[:, 0::2, 0::2]
                x = tf.contrib.layers.batch_norm(inputs=x, decay=0.9, scale=True, epsilon=0.0001, is_training=self.training, fused=False, updates_collections=None)
                x = tf.nn.relu(x)
                x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2, padding='SAME')
                input_pre = x
                mask_x = mask_x[:, 0::2, 0::2]
                self.dense_channels += self.input_conv_filters
                dense_out = x
                #### flowing into dense blocks and transition_layer ####
    
            for i in range(self.blocks):
                for j in range(self.level):
                    with tf.variable_scope('block_%dlevel_%d'%(i, j)) as scope:
                        #### [1, 1] convolution part for bottleneck ####
                        limit = self.bound(self.dense_channels, 4 * self.growth_rate, [1,1])
                        x = tf.layers.conv2d(x, filters=4 * self.growth_rate, kernel_size=[1,1],
                            strides=1, padding='VALID', data_format='channels_last', use_bias=False, kernel_initializer=tf.random_uniform_initializer(-limit, limit, dtype=tf.float32))
                        x = tf.contrib.layers.batch_norm(inputs=x, decay=0.9, scale=True, epsilon=0.0001, is_training=self.training, fused=False, updates_collections=None)
                        x = tf.nn.relu(x)
                        x = tf.layers.dropout(inputs=x, rate=self.dropout_rate, training=self.training)
    
                        #### [3, 3] convolution part for regular convolve operation
                        limit = self.bound(4 * self.growth_rate, self.growth_rate, [3,3])
                        x = tf.layers.conv2d(x, filters=self.growth_rate, kernel_size=[3,3],
                            strides=1, padding='SAME', data_format='channels_last', use_bias=False, kernel_initializer=tf.random_uniform_initializer(-limit, limit, dtype=tf.float32))
                        x = tf.contrib.layers.batch_norm(inputs=x, decay=0.9, scale=True, epsilon=0.0001, is_training=self.training, fused=False, updates_collections=None)
                        x = tf.nn.relu(x)
                        x = tf.layers.dropout(inputs=x, rate=self.dropout_rate, training=self.training)
                        dense_out = tf.concat([dense_out, x], axis=3)
                        x = dense_out
                        #### calculate the filter number of dense block's output ####
                        self.dense_channels += self.growth_rate
    
                if i < self.blocks - 1:
                    with tf.variable_scope('transition_%d'%i) as scope:
                        compressed_channels = int(self.dense_channels * self.transition)
                        #### new dense channels for new dense block ####
                        self.dense_channels = compressed_channels
                        limit = self.bound(self.dense_channels, compressed_channels, [1,1])
                        x = tf.layers.conv2d(x, filters=compressed_channels, kernel_size=[1,1],
                            strides=1, padding='VALID', data_format='channels_last', use_bias=False, kernel_initializer=tf.random_uniform_initializer(-limit, limit, dtype=tf.float32))
                        x = tf.contrib.layers.batch_norm(inputs=x, decay=0.9, scale=True, epsilon=0.0001, is_training=self.training, fused=False, updates_collections=None)
                        x = tf.nn.relu(x)
                        x = tf.layers.dropout(inputs=x, rate=self.dropout_rate, training=self.training)
                        x = tf.layers.average_pooling2d(inputs=x, pool_size=[2,2], strides=2, padding='SAME')
                        dense_out = x
                        mask_x = mask_x[:, 0::2, 0::2]
    
            return dense_out, mask_x
    
    
    
    class Attender():
        def __init__(self, channels,                                # output of Watcher | [batch, h, w, channels]
                    dim_decoder, dim_attend):                       # decoder hidden state:$h_{t-1}$ | [batch, dec_dim]
    
            self.channels = channels
    
            self.coverage_kernel = [11,11]                          # kernel size of $Q$
            self.coverage_filters = dim_attend                      # filter numbers of $Q$ | 512
    
            self.dim_decoder = dim_decoder                          # 256
            self.dim_attend = dim_attend                            # unified dim of three parts calculating $e_ti$ i.e. $Q*beta_t$, $U_a * a_i$, $W_a x h_{t-1}$ | 512
            self.U_f_init = norm_weight(self.coverage_filters, self.dim_attend)
            self.U_f_b_init = np.zeros((self.dim_attend,)).astype('float32')
            self.U_a_init = norm_weight(self.channels, self.dim_attend)
            self.U_a_b_init = np.zeros((self.dim_attend,)).astype('float32')
            self.W_a_init = norm_weight(self.dim_decoder, self.dim_attend)
            self.W_a_b_init = np.zeros((self.dim_attend,)).astype('float32')
            self.V_a_init = norm_weight(self.dim_attend, 1)
            self.V_a_b_init = np.zeros((1,)).astype('float32')
            self.alpha_past_filter_init = conv_norm_weight(1, self.dim_attend, self.coverage_kernel)
    
            self.U_f = tf.get_variable(initializer=tf.constant_initializer(self.U_f_init), shape=self.U_f_init.shape ,name='U_f')          # $U_f x f_i$ | [cov_filters, dim_attend]
            self.U_f_b = tf.get_variable(initializer=tf.constant_initializer(self.U_f_b_init), shape=self.U_f_b_init.shape ,name='U_f_b')  # $U_f x f_i + U_f_b$ | [dim_attend, ]
            self.U_a = tf.get_variable(initializer=tf.constant_initializer(self.U_a_init), shape=self.U_a_init.shape ,name='U_a')          # $U_a x a_i$ | [annotatin_channels, dim_attend]
            self.U_a_b = tf.get_variable(initializer=tf.constant_initializer(self.U_a_b_init), shape=self.U_a_b_init.shape ,name='U_a_b')  # $U_a x a_i + U_a_b$ | [dim_attend, ]
            self.W_a = tf.get_variable(initializer=tf.constant_initializer(self.W_a_init), shape=self.W_a_init.shape ,name='W_a')          # $W_a x h_{t_1}$ | [dec_dim, dim_attend]
            self.W_a_b = tf.get_variable(initializer=tf.constant_initializer(self.W_a_b_init), shape=self.W_a_b_init.shape ,name='W_a_b')  # $W_a x h_{t-1} + W_a_b$ | [dim_attend, ]
            self.V_a = tf.get_variable(initializer=tf.constant_initializer(self.V_a_init), shape=self.V_a_init.shape ,name='V_a')          # $V_a x tanh(A + B + C)$ | [dim_attend, 1]
            self.V_a_b = tf.get_variable(initializer=tf.constant_initializer(self.V_a_b_init), shape=self.V_a_b_init.shape ,name='V_a_b')  # $V_a x tanh(A + B + C) + V_a_b$ | [1, ]
            self.alpha_past_filter = tf.get_variable(initializer=tf.constant_initializer(self.alpha_past_filter_init), shape=self.alpha_past_filter_init.shape ,name='alpha_past_filter')
    
    
        def get_context(self, annotation4ctx, h_t_1, alpha_past4ctx, a_mask):
            #### calculate $U_f x f_i$ ####
            alpha_past_4d = alpha_past4ctx[:, :, :, None]
    
            Ft = tf.nn.conv2d(alpha_past_4d, filter=self.alpha_past_filter, strides=[1, 1, 1, 1], padding='SAME')
    
            coverage_vector = tf.tensordot(Ft, self.U_f, axes=1) \
            + self.U_f_b                                            # [batch, h, w, dim_attend]
    
            #### calculate $U_a x a_i$ ####
            watch_vector = tf.tensordot(annotation4ctx, self.U_a, axes=1) \
            + self.U_a_b                                            # [batch, h, w, dim_attend]
    
            #### calculate $W_a x h_{t - 1}$ ####
            speller_vector = tf.tensordot(h_t_1, self.W_a, axes=1) \
            + self.W_a_b                                            # [batch, dim_attend]
            speller_vector = speller_vector[:, None, None, :]       # [batch, None, None, dim_attend]
    
            tanh_vector = tf.tanh(
                coverage_vector + watch_vector + speller_vector)    # [batch, h, w, dim_attend]
    
            e_ti = tf.tensordot(tanh_vector, self.V_a, axes=1) + self.V_a_b  # [batch, h, w, 1]
    
            alpha = tf.exp(e_ti)
    
            alpha = tf.squeeze(alpha, axis=3)
    
            if a_mask is not None:
                alpha = alpha * a_mask
    
            alpha = alpha / tf.reduce_sum(alpha, \
                axis=[1, 2], keepdims=True)                         # normlized weights | [batch, h, w]
    
            alpha_past4ctx += alpha                                 # accumalated weights matrix | [batch, h, w]
    
            context = tf.reduce_sum(annotation4ctx * alpha[:, :, :, None], \
                axis=[1, 2])                                        # context vector | [batch, feature_channels]
    
            return context, alpha, alpha_past4ctx
    
    
    class Parser():
    
        def __init__(self, hidden_dim, word_dim, attender, context_dim):
    
            self.attender = attender                                # inner-instance of Attender to provide context
            self.context_dim = context_dim                          # context dime 684
            self.hidden_dim = hidden_dim                            # dim of hidden state  256
            self.word_dim = word_dim                                # dim of embedding word 256
    
            self.W_yz_yr_init = np.concatenate(
                [norm_weight(self.word_dim, self.hidden_dim), norm_weight(self.word_dim, self.hidden_dim)], axis=1)
            self.b_yz_yr_init = np.zeros((2 * self.hidden_dim, )).astype('float32')
            self.U_hz_hr_init = np.concatenate(
                [ortho_weight(self.hidden_dim),ortho_weight(self.hidden_dim)], axis=1)
    
            self.W_yh_init = norm_weight(self.word_dim, self.hidden_dim)
            self.b_yh_init = np.zeros((self.hidden_dim, )).astype('float32')
            self.U_rh_init = ortho_weight(self.hidden_dim)
            self.U_hz_hr_nl_init = np.concatenate(
                [ortho_weight(self.hidden_dim), ortho_weight(self.hidden_dim)], axis=1)
            self.b_hz_hr_nl_init = np.zeros((2 * self.hidden_dim, )).astype('float32')
            self.W_c_z_r_init = norm_weight(self.context_dim,
                2 * self.hidden_dim)
            self.U_rh_nl_init = ortho_weight(self.hidden_dim)
            self.b_rh_nl_init = np.zeros((self.hidden_dim, )).astype('float32')
            self.W_c_h_nl_init = norm_weight(self.context_dim, self.hidden_dim)
    
    
            self.W_yz_yr = tf.get_variable(initializer=tf.constant_initializer(self.W_yz_yr_init), shape=self.W_yz_yr_init.shape, name='W_yz_yr')             # [dim_word, 2 * dim_decoder]
            self.b_yz_yr = tf.get_variable(initializer=tf.constant_initializer(self.b_yz_yr_init), shape=self.b_yz_yr_init.shape, name='b_yz_yr')
            self.U_hz_hr = tf.get_variable(initializer=tf.constant_initializer(self.U_hz_hr_init), shape=self.U_hz_hr_init.shape, name='U_hz_hr')             # [dim_hidden, 2 * dim_hidden]
            self.W_yh = tf.get_variable(initializer=tf.constant_initializer(self.W_yh_init), shape=self.W_yh_init.shape, name='W_yh')
            self.b_yh = tf.get_variable(initializer=tf.constant_initializer(self.b_yh_init), shape=self.b_yh_init.shape, name='b_yh')                         # [dim_decoder, ]
            self.U_rh = tf.get_variable(initializer=tf.constant_initializer(self.U_rh_init), shape=self.U_rh_init.shape, name='U_rh')                         # [dim_hidden, dim_hidden]
            self.U_hz_hr_nl = tf.get_variable(initializer=tf.constant_initializer(self.U_hz_hr_nl_init), shape=self.U_hz_hr_nl_init.shape, name='U_hz_hr_nl') # [dim_hidden, 2 * dim_hidden] non_linear
            self.b_hz_hr_nl = tf.get_variable(initializer=tf.constant_initializer(self.b_hz_hr_nl_init), shape=self.b_hz_hr_nl_init.shape, name='b_hz_hr_nl') # [2 * dim_hidden, ]
            self.W_c_z_r = tf.get_variable(initializer=tf.constant_initializer(self.W_c_z_r_init), shape=self.W_c_z_r_init.shape, name='W_c_z_r')
            self.U_rh_nl = tf.get_variable(initializer=tf.constant_initializer(self.U_rh_nl_init), shape=self.U_rh_nl_init.shape, name='U_rh_nl')
            self.b_rh_nl = tf.get_variable(initializer=tf.constant_initializer(self.b_rh_nl_init), shape=self.b_rh_nl_init.shape, name='b_rh_nl')
            self.W_c_h_nl = tf.get_variable(initializer=tf.constant_initializer(self.W_c_h_nl_init), shape=self.W_c_h_nl_init.shape, name='W_c_h_nl')
    
    
    
        def get_ht_ctx(self, emb_y, target_hidden_state_0, annotations, a_m, y_m):
            res = tf.scan(self.one_time_step, elems=(emb_y, y_m),
                initializer=(target_hidden_state_0,
                    tf.zeros([tf.shape(annotations)[0], self.context_dim]),
                    tf.zeros([tf.shape(annotations)[0], tf.shape(annotations)[1], tf.shape(annotations)[2]]),
                    tf.zeros([tf.shape(annotations)[0], tf.shape(annotations)[1], tf.shape(annotations)[2]]),
                    annotations, a_m))
            return res
    
    
    
    
        def one_time_step(self, tuple_h0_ctx_alpha_alpha_past_annotation, tuple_emb_mask):
    
            target_hidden_state_0 = tuple_h0_ctx_alpha_alpha_past_annotation[0]
            alpha_past_one        = tuple_h0_ctx_alpha_alpha_past_annotation[3]
            annotation_one        = tuple_h0_ctx_alpha_alpha_past_annotation[4]
            a_mask                = tuple_h0_ctx_alpha_alpha_past_annotation[5]
    
            emb_y, y_mask = tuple_emb_mask
    
            emb_y_z_r_vector = tf.tensordot(emb_y, self.W_yz_yr, axes=1) + \
            self.b_yz_yr                                            # [batch, 2 * dim_decoder]
            hidden_z_r_vector = tf.tensordot(target_hidden_state_0,
            self.U_hz_hr, axes=1)                                   # [batch, 2 * dim_decoder]
            pre_z_r_vector = tf.sigmoid(emb_y_z_r_vector + \
            hidden_z_r_vector)                                      # [batch, 2 * dim_decoder]
    
            r1 = pre_z_r_vector[:, :self.hidden_dim]                # [batch, dim_decoder]
            z1 = pre_z_r_vector[:, self.hidden_dim:]                # [batch, dim_decoder]
    
            emb_y_h_vector = tf.tensordot(emb_y, self.W_yh, axes=1) + \
            self.b_yh                                               # [batch, dim_decoder]
            hidden_r_h_vector = tf.tensordot(target_hidden_state_0,
            self.U_rh, axes=1)                                      # [batch, dim_decoder]
            hidden_r_h_vector *= r1
            pre_h_proposal = tf.tanh(hidden_r_h_vector + emb_y_h_vector)
    
            pre_h = z1 * target_hidden_state_0 + (1. - z1) * pre_h_proposal
    
            if y_mask is not None:
                pre_h = y_mask[:, None] * pre_h + (1. - y_mask)[:, None] * target_hidden_state_0
    
            context, alpha, alpha_past_one = self.attender.get_context(annotation_one, pre_h, alpha_past_one, a_mask)  # [batch, dim_ctx]
            emb_y_z_r_nl_vector = tf.tensordot(pre_h, self.U_hz_hr_nl, axes=1) + self.b_hz_hr_nl
            context_z_r_vector = tf.tensordot(context, self.W_c_z_r, axes=1)
            z_r_vector = tf.sigmoid(emb_y_z_r_nl_vector + context_z_r_vector)
    
            r2 = z_r_vector[:, :self.hidden_dim]
            z2 = z_r_vector[:, self.hidden_dim:]
    
            emb_y_h_nl_vector = tf.tensordot(pre_h, self.U_rh_nl, axes=1) + self.b_rh_nl
            emb_y_h_nl_vector *= r2
            context_h_vector = tf.tensordot(context, self.W_c_h_nl, axes=1)
            h_proposal = tf.tanh(emb_y_h_nl_vector + context_h_vector)
            h = z2 * pre_h + (1. - z2) * h_proposal
    
            if y_mask is not None:
                h = y_mask[:, None] * h + (1. - y_mask)[:, None] * pre_h
    
            return h, context, alpha, alpha_past_one, annotation_one, a_mask
    
    
    class WAP():
        def __init__(self, watcher, attender, parser, hidden_dim, word_dim, context_dim, target_dim):
    
            self.hidden_dim = hidden_dim
            self.word_dim = word_dim
            self.context_dim = context_dim
            self.target_dim = target_dim
    
            self.watcher = watcher
            self.attender = attender
            self.parser = parser
    
            self.embed_matrix_init = norm_weight(self.target_dim, self.word_dim)
            self.Wa2h_init = norm_weight(self.context_dim, self.hidden_dim)
            self.ba2h_init = np.zeros((self.hidden_dim,)).astype('float32')
            self.Wc_init = norm_weight(self.context_dim, self.word_dim)
            self.bc_init = np.zeros((self.word_dim,)).astype('float32')
            self.Wh_init = norm_weight(self.hidden_dim, self.word_dim)
            self.bh_init = np.zeros((self.word_dim,)).astype('float32')
            self.Wy_init = norm_weight(self.word_dim, self.word_dim)
            self.by_init = np.zeros((self.word_dim,)).astype('float32')
            self.Wo_init = norm_weight(self.word_dim//2, self.target_dim)
            self.bo_init = np.zeros((self.target_dim,)).astype('float32')
    
            self.embed_matrix = tf.get_variable(initializer=tf.constant_initializer(self.embed_matrix_init), shape=self.embed_matrix_init.shape, name='embed')
            self.Wa2h = tf.get_variable(initializer=tf.constant_initializer(self.Wa2h_init), shape=self.Wa2h_init.shape , name='Wa2h')
            self.ba2h = tf.get_variable(initializer=tf.constant_initializer(self.ba2h_init), shape=self.ba2h_init.shape , name='ba2h')
            self.Wc = tf.get_variable(initializer=tf.constant_initializer(self.Wc_init), shape=self.Wc_init.shape , name='Wc')
            self.bc = tf.get_variable(initializer=tf.constant_initializer(self.bc_init), shape=self.bc_init.shape , name='bc')
            self.Wh = tf.get_variable(initializer=tf.constant_initializer(self.Wh_init), shape=self.Wh_init.shape , name='Wh')
            self.bh = tf.get_variable(initializer=tf.constant_initializer(self.bh_init), shape=self.bh_init.shape , name='bh')
            self.Wy = tf.get_variable(initializer=tf.constant_initializer(self.Wy_init), shape=self.Wy_init.shape , name='Wy')
            self.by = tf.get_variable(initializer=tf.constant_initializer(self.by_init), shape=self.by_init.shape , name='by')
            self.Wo = tf.get_variable(initializer=tf.constant_initializer(self.Wo_init), shape=self.Wo_init.shape , name='Wo')
            self.bo = tf.get_variable(initializer=tf.constant_initializer(self.bo_init), shape=self.bo_init.shape , name='bo')
    
    
        def get_cost(self, cost_annotation, cost_y, a_m, y_m):
            timesteps = tf.shape(cost_y)[0]
            batch_size = tf.shape(cost_y)[1]
            emb_y = tf.nn.embedding_lookup(self.embed_matrix, tf.reshape(cost_y, [-1]))
            emb_y = tf.reshape(emb_y, [timesteps, batch_size, self.word_dim])
            emb_pad = tf.fill((1, batch_size, self.word_dim), 0.0)
            emb_shift = tf.concat([emb_pad ,tf.strided_slice(emb_y, [0, 0, 0], [-1, batch_size, self.word_dim], [1, 1, 1])], axis=0)
            new_emb_y = emb_shift
            anno_mean = tf.reduce_sum(cost_annotation * a_m[:, :, :, None], axis=[1, 2]) / tf.reduce_sum(a_m, axis=[1, 2])[:, None]
            h_0 = tf.tensordot(anno_mean, self.Wa2h, axes=1) + self.ba2h  # [batch, hidden_dim]
            h_0 = tf.tanh(h_0)
            ret = self.parser.get_ht_ctx(new_emb_y, h_0, cost_annotation, a_m, y_m)
            h_t = ret[0]                      # h_t of all timesteps [timesteps, batch, word_dim]
            c_t = ret[1]                      # c_t of all timesteps [timesteps, batch, context_dim]
            alpha_t = ret[2]
            alpha_past_t = ret[3]
            y_t_1 = new_emb_y                 # shifted y | [1:] = [:-1]
            logit_gru = tf.tensordot(h_t, self.Wh, axes=1) + self.bh
            logit_ctx = tf.tensordot(c_t, self.Wc, axes=1) + self.bc
            logit_pre = tf.tensordot(y_t_1, self.Wy, axes=1) + self.by
            logit = logit_pre + logit_ctx + logit_gru
            shape = tf.shape(logit)
            logit = tf.reshape(logit, [shape[0], -1, shape[2]//2, 2])
            logit = tf.reduce_max(logit, axis=3)
            logit = tf.tensordot(logit, self.Wo, axes=1) + self.bo
            logit_shape = tf.shape(logit)
            logit = tf.reshape(logit, [-1,
                logit_shape[2]])
            cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=tf.one_hot(tf.reshape(cost_y, [-1]),
                depth=self.target_dim))
    
            cost = tf.multiply(cost, tf.reshape(y_m, [-1]))
            cost = tf.reshape(cost, [shape[0], shape[1]])
            cost = tf.reduce_sum(cost, axis=0)
            cost = tf.reduce_mean(cost)
            return cost
    
    
    
        def get_word(self, sample_y, sample_h_pre, alpha_past_pre, sample_annotation):
    
            emb = tf.cond(sample_y[0] < 0,
                lambda: tf.fill((1, self.word_dim), 0.0),
                lambda: tf.nn.embedding_lookup(wap.embed_matrix, sample_y)
                )
    
            emb_y_z_r_vector = tf.tensordot(emb, self.parser.W_yz_yr, axes=1) + \
            self.parser.b_yz_yr                                            # [batch, 2 * dim_decoder]
            hidden_z_r_vector = tf.tensordot(sample_h_pre,
            self.parser.U_hz_hr, axes=1)                                   # [batch, 2 * dim_decoder]
            pre_z_r_vector = tf.sigmoid(emb_y_z_r_vector + \
            hidden_z_r_vector)                                             # [batch, 2 * dim_decoder]
    
            r1 = pre_z_r_vector[:, :self.parser.hidden_dim]                # [batch, dim_decoder]
            z1 = pre_z_r_vector[:, self.parser.hidden_dim:]                # [batch, dim_decoder]
    
            emb_y_h_vector = tf.tensordot(emb, self.parser.W_yh, axes=1) + \
            self.parser.b_yh                                               # [batch, dim_decoder]
            hidden_r_h_vector = tf.tensordot(sample_h_pre,
            self.parser.U_rh, axes=1)                                      # [batch, dim_decoder]
            hidden_r_h_vector *= r1
            pre_h_proposal = tf.tanh(hidden_r_h_vector + emb_y_h_vector)
    
            pre_h = z1 * sample_h_pre + (1. - z1) * pre_h_proposal
    
            context, alpha, alpha_past = self.parser.attender.get_context(sample_annotation, pre_h, alpha_past_pre, None)  # [batch, dim_ctx]
            emb_y_z_r_nl_vector = tf.tensordot(pre_h, self.parser.U_hz_hr_nl, axes=1) + self.parser.b_hz_hr_nl
            context_z_r_vector = tf.tensordot(context, self.parser.W_c_z_r, axes=1)
            z_r_vector = tf.sigmoid(emb_y_z_r_nl_vector + context_z_r_vector)
    
            r2 = z_r_vector[:, :self.parser.hidden_dim]
            z2 = z_r_vector[:, self.parser.hidden_dim:]
    
            emb_y_h_nl_vector = tf.tensordot(pre_h, self.parser.U_rh_nl, axes=1) + self.parser.b_rh_nl
            emb_y_h_nl_vector *= r2
            context_h_vector = tf.tensordot(context, self.parser.W_c_h_nl, axes=1)
            h_proposal = tf.tanh(emb_y_h_nl_vector + context_h_vector)
            h = z2 * pre_h + (1. - z2) * h_proposal
    
            h_t = h
            c_t = context
            alpha_t = alpha
            alpha_past_t = alpha_past
            y_t_1 = emb
            logit_gru = tf.tensordot(h_t, self.Wh, axes=1) + self.bh
            logit_ctx = tf.tensordot(c_t, self.Wc, axes=1) + self.bc
            logit_pre = tf.tensordot(y_t_1, self.Wy, axes=1) + self.by
            logit = logit_pre + logit_ctx + logit_gru   # batch x word_dim
    
            shape = tf.shape(logit)
            logit = tf.reshape(logit, [-1, shape[1]//2, 2])
            logit = tf.reduce_max(logit, axis=2)
            logit = tf.tensordot(logit, self.Wo, axes=1) + self.bo
    
            next_probs = tf.nn.softmax(logits=logit)
            next_word  = tf.reduce_max(tf.multinomial(next_probs, num_samples=1), axis=1)
            return next_probs, next_word, h_t, alpha_past_t
    
        def get_sample(self, p, w, h, alpha, ctx0, h_0, k , maxlen, stochastic, a_h, a_w, session):
    
            sample = []
            sample_score = []
    
            live_k = 1
            dead_k = 0
    
            hyp_samples = [[]] * live_k
            hyp_scores = np.zeros(live_k).astype('float32')
            hyp_states = []
    
    
            next_alpha_past = np.zeros((ctx0.shape[0], a_h, a_w)).astype('float32')
            emb_0 = np.zeros((ctx0.shape[0], 256))
    
            next_w = -1 * np.ones((1,)).astype('int64')
    
            next_state = h_0
            for ii in range(maxlen):
    
                ctx = np.tile(ctx0, [live_k, 1, 1, 1])
    
                input_dict = {
                anno:ctx,
                infer_y:next_w,
                alpha_past:next_alpha_past,
                h_pre:next_state
                }
    
                next_p, next_w, next_state, next_alpha_past = session.run([p, w, h, alpha], feed_dict=input_dict)
    
                if stochastic:
                    if argmax:
                        nw = next_p[0].argmax()
                    else:
                        nw = next_w[0]
                    sample.append(nw)
                    sample_score += next_p[0, nw]
                    if nw == 0:
                        break
                else:
                    cand_scores = hyp_scores[:, None] - np.log(next_p)
                    cand_flat = cand_scores.flatten()
                    ranks_flat = cand_flat.argsort()[:(k-dead_k)]
                    voc_size = next_p.shape[1]
    
                    assert voc_size==111
    
                    trans_indices = ranks_flat // voc_size
                    word_indices = ranks_flat % voc_size
                    costs = cand_flat[ranks_flat]
                    new_hyp_samples = []
                    new_hyp_scores = np.zeros(k-dead_k).astype('float32')
                    new_hyp_states = []
                    new_hyp_alpha_past = []
    
                    for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                        new_hyp_samples.append(hyp_samples[ti]+[wi])
                        new_hyp_scores[idx] = copy.copy(costs[idx])
                        new_hyp_states.append(copy.copy(next_state[ti]))
                        new_hyp_alpha_past.append(copy.copy(next_alpha_past[ti]))
    
                    new_live_k = 0
                    hyp_samples = []
                    hyp_scores = []
                    hyp_states = []
                    hyp_alpha_past = []
    
                    for idx in range(len(new_hyp_samples)):
                        if new_hyp_samples[idx][-1] == 0: # <eol>
                            sample.append(new_hyp_samples[idx])
                            sample_score.append(new_hyp_scores[idx])
                            dead_k += 1
                        else:
                            new_live_k += 1
                            hyp_samples.append(new_hyp_samples[idx])
                            hyp_scores.append(new_hyp_scores[idx])
                            hyp_states.append(new_hyp_states[idx])
                            hyp_alpha_past.append(new_hyp_alpha_past[idx])
                    hyp_scores = np.array(hyp_scores)
                    live_k = new_live_k
    
                    if new_live_k < 1:
                        break
                    if dead_k >= k:
                        break
    
                    next_w = np.array([w[-1] for w in hyp_samples])
                    next_state = np.array(hyp_states)
                    next_alpha_past = np.array(hyp_alpha_past)
    
            if not stochastic:
                # dump every remaining one
                if live_k > 0:
                    for idx in range(live_k):
                        sample.append(hyp_samples[idx])
                        sample_score.append(hyp_scores[idx])
    
            return sample, sample_score


#################################################################### model above ####################################################################

    if args.download_only:
        sys.exit(0)

    if args.job_name is None or args.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")
    if args.task_index is None or args.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")

    print("job name = %s" % args.job_name)
    print("task index = %d" % args.task_index)

    # Construct the cluster and start the server
    ps_spec = args.ps_hosts.split(",")
    worker_spec = args.worker_hosts.split(",")

    # Get the number of workers.
    num_workers = len(worker_spec)
    num_pss = len(ps_spec)

    cluster = tf.train.ClusterSpec({
        "ps": ps_spec,
        "worker": worker_spec})

    if not args.existing_servers:
        # Not using existing servers. Create an in-process server.
        server = tf.train.Server(
            cluster, job_name=args.job_name, task_index=args.task_index, protocol=args.protocol)
        if args.job_name == "ps":
            
            config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            sess = tf.Session(server.target, config=config)

            print("ps %d, create done queue" % args.task_index)
            queue = create_done_queue(args.task_index, num_workers)

            print("ps %d, running" % args.task_index)
            for i in range(num_workers):
                sess.run(queue.dequeue())
                print("ps %d received worker %d done" % (args.task_index, i))

            print("all workers are done, ps %d: exit" % (args.task_index))
            sys.exit()

    is_chief = (args.task_index == 0)
    if args.num_gpus > 0:
        # Avoid gpu allocation conflict: now allocate task_num -> #gpu
        # for each worker in the corresponding machine
        gpu = (args.task_index % args.num_gpus)
        worker_device = "/job:worker/task:%d/gpu:%d" % (args.task_index, gpu)
    elif args.num_gpus == 0:
        # Just allocate the CPU to worker server
        cpu = 0
        worker_device = "/job:worker/task:%d/cpu:%d" % (args.task_index, cpu)

    print("worker %d, worker_device=%s" % (args.task_index, worker_device))
    print("worker %d, create done queue" % args.task_index)
    queues = create_done_queues(num_pss, num_workers)
    print("worker %d, done queue created" % args.task_index)

    # The device setter will automatically place Variables ops on separate
    # parameter servers (ps). The non-Variable ops will be placed on the workers.
    # The ps use CPU and workers use corresponding GPU
    device_setter = tf.train.replica_device_setter(
                worker_device=worker_device,
                ps_device="/job:ps/cpu:0",
                cluster=cluster)
    with tf.device(device_setter):
        global_step = tf.train.get_or_create_global_step()
        x = tf.placeholder_with_default(input=np.zeros(shape=(1, 1, 1, 1), dtype=np.float32), shape=[None, None, None, 1])
        y = tf.placeholder_with_default(input=np.zeros(shape=(1, 1), dtype=np.int32), shape=[None, None])
        x_mask = tf.placeholder_with_default(input=np.zeros(shape=(1, 1, 1), dtype=np.float32), shape=[None, None, None])
        y_mask = tf.placeholder_with_default(input=np.zeros(shape=(1, 1), dtype=np.float32), shape=[None, None])
        lr = tf.placeholder_with_default(input=args.learning_rate, shape=())
        if_trainning = tf.placeholder_with_default(input=False, shape=())
        watcher_train = Watcher_train(blocks=3,level=16, growth_rate=24, training=if_trainning)
        annotation, anno_mask = watcher_train.dense_net(x, x_mask)
        anno = tf.placeholder(tf.float32)
        infer_y = tf.placeholder(tf.int64)
        h_pre = tf.placeholder(tf.float32)
        alpha_past = tf.placeholder(tf.float32)
        attender = Attender(annotation.shape.as_list()[3], 256, 512)
        parser = Parser(256, 256, attender, annotation.shape.as_list()[3])
        wap = WAP(watcher_train, attender, parser, 256, 256, annotation.shape.as_list()[3], 111)
        hidden_state_0 = tf.tanh(tf.tensordot(tf.reduce_mean(anno, axis=[1, 2]), wap.Wa2h, axes=1) + wap.ba2h)  # [batch, hidden_dim]
        cost = wap.get_cost(annotation, y, anno_mask, y_mask)

        vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for vv in vs:
            if not 'BatchNorm' in vv.name:
                print('regularized ', vv.name)
                cost += 1e-4 * tf.reduce_sum(tf.pow(vv, 2))
                
        p, w, h, alpha = wap.get_word(infer_y, h_pre, alpha_past, anno)

        opt = tf.train.AdadeltaOptimizer(learning_rate=lr)

        if args.sync_replicas:
            if args.replicas_to_aggregate is None:
                replicas_to_aggregate = num_workers
            else:
                replicas_to_aggregate = args.replicas_to_aggregate

            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=replicas_to_aggregate,
                total_num_replicas=num_workers,
                name="mnist_sync_replicas")

        sync_replicas_hook = opt.make_session_run_hook(is_chief, num_tokens=0)
        train_step = opt.minimize(cost, global_step=global_step)

        enq_ops = []
        for q in queues:
            qop = q.enqueue(1)
            enq_ops.append(qop)

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps", "/job:worker/task:%d" % args.task_index])

    if args.infer_shapes == True:
        sess_config.graph_options.infer_shapes = args.infer_shapes


    # The chief worker (task_index==0) session will prepare the session,
    # while the remaining workers will wait for the preparation to complete.
    if is_chief:
        print("Worker %d: Initializing session..." % args.task_index)
    else:
        print("Worker %d: Waiting for session to be initialized..." %
              args.task_index)
    
    if args.existing_servers:
        server_grpc_url = "grpc://" + worker_spec[args.task_index]
        print("Using existing server at: %s" % server_grpc_url)

        sess = sv.prepare_or_wait_for_session(server_grpc_url,
                                              config=sess_config)
    else:
        hooks=[sync_replicas_hook, tf.train.StopAtStepHook(last_step=10000000)]
        sess = tf.train.MonitoredTrainingSession(master=server.target,
            is_chief=is_chief,
            hooks=hooks,
            save_checkpoint_secs=60,
            config=sess_config)
    

    print("Worker %d: Session initialization complete." % args.task_index)

    # Perform training
    time_begin = time.time()
    print("Training begins @ %f" % time_begin)

    worddicts = load_dict(args.path + '/data/dictionary.txt')
    worddicts_r = [None] * len(worddicts)
    for kk, vv in worddicts.items():
      worddicts_r[vv] = kk

    train, train_uid_list = dataIterator(args.path + 'data/offline-train.pkl', args.path + 'data/train_caption.txt', worddicts,
      batch_size=args.batch_size, batch_Imagesize=400000, maxlen=100, maxImagesize=400000)

    valid, valid_uid_list = dataIterator(args.path + 'data/offline-test.pkl', args.path + 'data/test_caption.txt', worddicts,
      batch_size=args.batch_size, batch_Imagesize=400000, maxlen=100, maxImagesize=400000)

    quota = len(train) // args.num_gpus

    print('train lenght is ', quota)

    local_step = 0
    step = 0
    cost_s = 0
    dispFreq = quota
    saveFreq = len(train)
    sampleFreq = len(train)
    validFreq = len(train)
    history_errs = []
    estop = False
    halfLrFlag = 0
    patience = 30
    lrate = args.learning_rate
    max_epoch = 800
    valid_cost = 1000
    decode = False

    for epoch in range(max_epoch):

        random.shuffle(train)
        begin = (args.task_index) % args.num_gpus

        train_i = train[quota * begin: quota * (begin + 1)]

        for batch_x, batch_y in train_i:

            batch_x, batch_x_m, batch_y, batch_y_m = prepare_data(batch_x, batch_y)

            train_feed = {x: batch_x, y: batch_y, x_mask:batch_x_m, y_mask:batch_y_m, if_trainning:True, lr:lrate}
    
            cost_i, _, step = sess.run([cost, train_step, global_step], feed_dict=train_feed)
            local_step += 1
            cost_s += cost_i
    
        now = time.time()

        print("%f: Worker %d: training step %d done (global step: %d) cost is %3.2f learning rate is %f epoch is %d" % (now, args.task_index, local_step, step, cost_s/dispFreq, lrate, epoch))
        cost_s = 0

        if is_chief:
            decode = True
            fpp_sample = open(args.path + 'result/valid_decode_result_task_%d_%s.txt'%(args.task_index, args.token), 'w')
            valid_count_idx = 0
            for batch_x, batch_y in valid:
                for xx in batch_x:
                    xx = np.moveaxis(xx, 0, -1)
                    xx_pad = np.zeros((xx.shape[0], xx.shape[1], xx.shape[2]), dtype='float32')
                    xx_pad[:,:, :] = xx / 255.
                    xx_pad = xx_pad[None, :, :, :]
                    annot = sess.run(annotation, feed_dict={x:xx_pad, if_trainning:False})
                    h_state = sess.run(hidden_state_0, feed_dict={anno:annot})
                    sample, score = wap.get_sample(p, w, h, alpha, annot, h_state,
                     10, 100, False, annot.shape[1], annot.shape[2], sess)
                    score = score / np.array([len(s) for s in sample])
                    ss = sample[score.argmin()]
                    fpp_sample.write(valid_uid_list[valid_count_idx])
                    valid_count_idx=valid_count_idx+1
                    if np.mod(valid_count_idx, 100) == 0:
                        print('gen %d samples'%valid_count_idx)
                    for vv in ss:
                        if vv == 0: # <eol>
                            break
                        fpp_sample.write(' '+worddicts_r[vv])
                    fpp_sample.write('\n')
            fpp_sample.close()
            print('valid set decode done')
    
        if is_chief:
            probs = []
            for batch_x, batch_y in valid:
                batch_x, batch_x_m, batch_y, batch_y_m = prepare_data(batch_x, batch_y)
                pprobs = sess.run(cost, feed_dict={x:batch_x, y:batch_y, x_mask:batch_x_m, y_mask:batch_y_m, if_trainning:False})
                #print(pprobs)
                probs.append(pprobs)
            valid_errs = np.array(probs)
            valid_err_cost = valid_errs.mean()
            if valid_cost > valid_err_cost:
                valid_cost = valid_err_cost
            print('validation cost is ', valid_err_cost)
            if decode:
                process(args.path + 'result/valid_decode_result_task_%d_%s.txt'%(args.task_index, args.token), args.path + 'data/test_caption.txt',\
                    args.path + 'result/valid_task_%d_%s.wer'%(args.task_index, args.token))
                fpp=open(args.path + 'result/valid_task_%d_%s.wer'%(args.task_index, args.token))
                stuff=fpp.readlines()
                fpp.close()
                m=re.search('WER (.*)\n',stuff[0])
                valid_per=100. * float(m.group(1))
                m=re.search('ExpRate (.*)\n',stuff[1])
                valid_sacc=100. * float(m.group(1))
                valid_err=valid_per
                history_errs.append(valid_err)
                if local_step/validFreq == 0 or valid_err <= np.array(history_errs).min():
                    bad_counter = 0
                if local_step/validFreq != 0 and valid_err > np.array(history_errs).min():
                    bad_counter += 1
                    if bad_counter > patience:
                        if halfLrFlag==2:
                            print('Early Stop!')
                            estop = True
                            break
                        else:
                            print('Lr decay and retrain!')
                            bad_counter = 0
                            lrate = lrate / 10
                            halfLrFlag += 1
                print('Valid WER: %.2f%%, ExpRate: %.2f%%, Cost: %f' % (valid_per,valid_sacc,valid_err_cost))
        if estop:
            break

        if step >= args.train_steps:
            break

    time_end = time.time()
    print("Training ends @ %f" % time_end)
    training_time = time_end - time_begin
    print("Training elapsed time: %f s" % training_time)
    for op in enq_ops:
        sess.run(op)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_only", type=bool, default=False)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--task_index", type=int)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--replicas_to_aggregate", type=int)
    parser.add_argument("--hidden_units", type=int, default=100)
    parser.add_argument("--train_steps", type=int, default=200000)
    parser.add_argument("--learning_rate", type=float, default=1.0)
    parser.add_argument("--sync_replicas", type=bool, default=True)
    parser.add_argument("--existing_servers", type=bool, default=False)
    parser.add_argument("--ps-hosts", default="localhost:2222")
    parser.add_argument("--ps_hosts", default="localhost:2222")
    parser.add_argument("--worker-hosts", default="localhost:2223,localhost:2224")
    parser.add_argument("--worker_hosts", default="localhost:2223,localhost:2224")
    parser.add_argument("--job-name")
    parser.add_argument("--job_name")
    parser.add_argument("--protocol", default="grpc")
    parser.add_argument("--infer_shapes", type=bool, default=False)
    parser.add_argument("--path", default="./")
    parser.add_argument("--token", default="")
    parser.add_argument("--batch_size", type=int, default=4)
    (args, unknown) = parser.parse_known_args()
    main(args)