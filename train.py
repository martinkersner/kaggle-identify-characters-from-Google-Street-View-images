#!/usr/bin/python

# Martin Kersner, m.kersner@gmail.com
# 2015/11/23

import os
import csv
import numpy as np
import label_init as li
import tools as tl
tl.load_caffe()
import caffe

#db_names        = ["data/train_lmdb",   "data/val_lmdb"]
#db_names        = ["data/train_256_lmdb",   "data/val_256_lmdb"]
db_names        = ["data/train_256_18849_lmdb",   "data/val_256_18849_lmdb"]
#input_img_lists = ["data/train.csv",    "data/val.csv"]
input_img_lists = ["data/train_18849.csv",    "data/val_18849.csv"]
#input_img_dirs  = ["data/trainResized", "data/trainResized"]
#input_img_dirs  = ["data/train_png_resized256", "data/train_png_resized256"]
input_img_dirs  = ["data/train_png_resized256_18849", "data/train_png_resized256_18849"]

#db_names        = ["data/train_val_lmdb"]
#input_img_lists = ["data/train_val.csv"]
#input_img_dirs  = ["data/trainResized"]

# Data preparation
for db, input_list, input_dir in zip(db_names, input_img_lists, input_img_dirs):
    img_names, img_labels = tl.read_img_names_from_csv(input_list, skip_header=False, delimiter=",", append_string=".png")

    img_idxs = [li.labels[cls] for cls in img_labels]

    if not tl.exist_dir(db):
        print "Creating " + db + " lmdb database."
        tl.imgs_to_lmdb(input_dir, img_names, db, labels=img_idxs)
    else:
        print "Database " + db + " already exists!"

# TODO create as parameters of script
n_iter = 2
net_name = "bvlc_reference_caffenet"

# Training
solver_path = os.path.join(net_name, "solver.prototxt")
model_path  = os.path.join(net_name, net_name + ".caffemodel")

caffe.set_mode_gpu()
solver = caffe.SGDSolver(solver_path)
solver.net.copy_from(model_path)
# TODO add condition for fine-tuning

train_loss = np.zeros(n_iter)
for it in range(n_iter):
    solver.step(1)
    train_loss[it] = solver.net.blobs['loss'].data
