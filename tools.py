#!/usr/bin/python

# Martin Kersner, m.kersner@gmail.com
# 2015/11/20

import os
import sys
import re
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

def generate_unique_id():
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

def create_log_dir(model_log_dir, log_id):
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

def modify_solver_parameter(solver_in, solver_out, field, value):
    with open(solver_in, 'rb') as f_in:
        with open(solver_out, 'wb') as f_out:
            for line in f_in:

                if (re.match(field, line)):
                    solver_write_field_value(f_out, field, str(value))
                else:
                    f_out.write(line)

def solver_write_field_value(f, field, value):
    if value.isdigit():
        f.write(field + ": " + value + "\n")
    else:
        f.write(field + ": \"" + value + "\"\n")

def log_to_file(log_dir, file_name, content):
    file_path = os.path.join(log_dir, file_name)

    with open(file_path, 'wb') as f:
        f.write(str(content))

    return content 

'''
Eli Bendersky
http://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
'''
from contextlib import contextmanager
import ctypes
import io
import tempfile

libc = ctypes.CDLL(None)
c_stderr = ctypes.c_void_p.in_dll(libc, 'stderr')

@contextmanager
def redirect_caffe_log(stream):
    # The original fd stderr points to.
    original_stderr_fd = sys.stderr.fileno()

    def _redirect_stderr(to_fd):
        """Redirect stderr to the given file descriptor."""
        # Flush the C-level buffer stderr
        libc.fflush(c_stderr)
        # Flush and close sys.stderr - also closes the file descriptor (fd)
        sys.stderr.close()
        # Make original_stderr_fd point to the same file as to_fd
        os.dup2(to_fd, original_stderr_fd)
        # Create a new sys.stderr that points to the redirected fd
        sys.stderr = os.fdopen(original_stderr_fd, 'wb')
    
    # Save a copy of the original stdout fd in saved_stderr_fd
    saved_stderr_fd = os.dup(original_stderr_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stderr(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stderr(saved_stderr_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(unicode(tfile.read(), 'unicode-escape'))
    finally:
        tfile.close()
        os.close(saved_stderr_fd)
