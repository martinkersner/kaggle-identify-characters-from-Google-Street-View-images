#!/usr/bin/python

# Martin Kersner, m.kersner@gmail.com
# 2015/11/23

import os
import csv
import label_init as li
import tools as tl
tl.load_caffe()
import caffe

#db_names        = ["data/train_lmdb",   "data/val_lmdb"]
#input_img_lists = ["data/train.csv",    "data/val.csv"]
#input_img_dirs  = ["data/trainResized", "data/trainResized"]

db_names        = ["data/train_val_lmdb"]
input_img_lists = ["data/train_val.csv"]
input_img_dirs  = ["data/trainResized"]

# Data preparation
for db, input_list, input_dir in zip(db_names, input_img_lists, input_img_dirs):
    img_names, img_labels = tl.read_img_names_from_csv(input_list, skip_header=False, delimiter=",", append_string=".Bmp")

    img_idxs = [li.labels[cls] for cls in img_labels]

    if not tl.exist_dir(db):
        print "Creating " + db + " lmdb database."
        tl.imgs_to_lmdb(input_dir, img_names, db, labels=img_idxs)
    else:
        print "Database " + db + " already exists!"

# Training using caffe
net_name = "lenet"
solver_path = os.path.join(net_name, "solver.prototxt")
caffe.set_mode_gpu()
solver = caffe.SGDSolver(solver_path)
solver.step(10000)
