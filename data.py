import numpy
import sys
import pickle as pkl
import gzip


def dataIterator(feature_file,label_file,dictionary,batch_size,batch_Imagesize,maxlen,maxImagesize):
    
    fp=open(feature_file,'rb')
    features=pkl.load(fp)
    fp.close()

    fp2=open(label_file,'r')
    labels=fp2.readlines()
    fp2.close()

    targets={}
    # map word to int with dictionary
    for l in labels:
        tmp=l.strip().split()
        uid=tmp[0]
        w_list=[]
        for w in tmp[1:]:
            if w in dictionary:
                w_list.append(dictionary[w])
            else:
                print('a word not in the dictionary !! sentence ',uid,'word ', w)
                sys.exit()
        targets[uid]=w_list



    imageSize={}
    for uid,fea in features.items():
        imageSize[uid]=fea.shape[1]*fea.shape[2]

    imageSize= sorted(imageSize.items(), key=lambda d:d[1]) # sorted by sentence length,  return a list with each triple element


    feature_batch=[]
    label_batch=[]
    feature_total=[]
    label_total=[]
    uidList=[]

    batch_image_size=0
    biggest_image_size=0
    i=0
    for uid,size in imageSize:
        if size>biggest_image_size:
            biggest_image_size=size
        fea=features[uid]
        lab=targets[uid]
        batch_image_size=biggest_image_size*(i+1)
        if len(lab)>maxlen:
            print('sentence', uid, 'length bigger than', maxlen, 'ignore')
        elif size>maxImagesize:
            print('image', uid, 'size bigger than', maxImagesize, 'ignore')
        else:
            uidList.append(uid)
            if batch_image_size>batch_Imagesize or i==batch_size: # a batch is full
                feature_total.append(feature_batch)
                label_total.append(label_batch)

                i=0
                biggest_image_size=size
                feature_batch=[]
                label_batch=[]
                feature_batch.append(fea)
                label_batch.append(lab)
                batch_image_size=biggest_image_size*(i+1)
                i+=1
            else:
                feature_batch.append(fea)
                label_batch.append(lab)
                i+=1

    # last batch
    feature_total.append(feature_batch)
    label_total.append(label_batch)

    print('total ',len(feature_total), 'batch data loaded')

    return list(zip(feature_total,label_total)),uidList


def load_dict(dictFile):
    fp=open(dictFile)
    stuff=fp.readlines()
    fp.close()
    lexicon={}
    for l in stuff:
        w=l.strip().split()
        lexicon[w[0]]=int(w[1])

    print('total words/phones',len(lexicon))
    return lexicon


def prepare_data(images_x, seqs_y, n_words_src=30000,
                 n_words=30000):

    heights_x = [s.shape[1] for s in images_x]
    widths_x = [s.shape[2] for s in images_x]
    lengths_y = [len(s) for s in seqs_y]

    n_samples = len(heights_x)
    max_height_x = numpy.max(heights_x)
    max_width_x = numpy.max(widths_x)
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((n_samples, max_height_x, max_width_x, 1)).astype('float32')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64') # the <eol> must be 0 in the dict !!!
    x_mask = numpy.zeros((n_samples, max_height_x, max_width_x)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(images_x, seqs_y)):
        x[idx, :heights_x[idx], :widths_x[idx], :] = (numpy.moveaxis(s_x, 0, -1) / 255.) # [B, C, H, W] -> [B, H, W, C]
        x_mask[idx, :heights_x[idx], :widths_x[idx]] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask
