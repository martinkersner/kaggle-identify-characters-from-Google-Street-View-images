#!/usr/bin/python

# Martin Kersner, m.kersner@gmail.com
# 2015/11/20

import os
import sys
import time
import csv
import json
import Image
import numpy as np
import lmdb
import shutil
import caffe

def save_image_from_arary(img_arr, img_name):
    im = Image.fromarray(img_arr)
    im.convert('RGB').save(img_name)

def load_caffe():
    caffe_root = os.environ['CAFFE_ROOT']
    sys.path.insert(0, caffe_root + '/python')

def normalize_to_range(img, max_thresh):
    img_min = np.min(img),
    img_max = np.max(img)
    diff = img_max - img_min

    return ((img - img_min) / diff) * max_thresh

def imgs_to_lmdb(path_src, src_imgs, path_dst, labels=None):
    '''
    Generate LMDB file from set of images
    Source: https://github.com/BVLC/caffe/issues/1698#issuecomment-70211045
    credit: Evan Shelhamer
    '''

    caffe.set_mode_gpu()

    if (labels == None):
        labels = [0] * len(src_imgs)

    db = lmdb.open(path_dst, map_size=int(1e12))

    with db.begin(write=True) as in_txn:
        for idx, img_name in enumerate(src_imgs):

            path_ = os.path.join(path_src, img_name)

            img = np.array(Image.open(path_).convert('RGB')).astype("uint8")
            img = img[:,:,::-1]
            img = img.transpose((2,0,1))
            img_dat = caffe.io.array_to_datum(img, labels[idx])
            in_txn.put('{:0>10d}'.format(idx), img_dat.SerializeToString())

    db.close()

    return 0

def exist_dir(dir_name):
    return os.path.isdir(dir_name)

def read_img_names_from_csv(file_name, skip_header=True, delimiter=" ", append_string=""):
    img_names  = []
    img_labels = []

    with open(file_name, 'rb') as f:
        reader = csv.reader(f, delimiter=delimiter)

        if (skip_header):
            next(reader, None)

        for row in reader:
            img_names.append(row[0] + append_string)
            img_labels.append(row[1])

    return img_names, img_labels

def current_time():
    return time.strftime("%Y-%m-%d-%H-%M-%S")

def tuple_unicode2str(key, value):
    if isinstance(key, unicode):
        key = str(key)
    if isinstance(value, unicode):
        value = str(value)

        # conversion to boolean 
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False 

    return (key, value)

def dict_unicode2str(dictionary):
    return dict([tuple_unicode2str(k, v) for k, v in dictionary.items()])

def load_settings(settings_filename):
    try:
        with open(settings_filename) as settings_file:    
            # caffe requires string type of str and not unicode
            settings = dict_unicode2str(json.load(settings_file))
            settings["settings_filename"] = settings_filename
    except IOError as e:
        print "Unable to open settings file!"
        exit()

    return settings

def check_arguments(argv, count, output_str):
    if (len(argv) != count+1):
        print output_str
        exit()

def create_log_dir(model_log_dir):
    log_id = current_time()
    log_dir = os.path.join(model_log_dir, log_id)
    os.makedirs(log_dir)

    return log_dir

def log_file(file_src_path, log_dir):
    shutil.copy(file_src_path, log_dir)

def sec2hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    hms_str = "%02d:%02d:%02d" % (h, m, s)

    return hms_str
