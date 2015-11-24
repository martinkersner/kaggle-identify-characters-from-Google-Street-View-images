#!/usr/bin/python

# Martin Kersner, m.kersner@gmail.com
# 2015/11/23

import os
import time
import csv
import tools as tl
tl.load_caffe()
import caffe

# Initialization
labels = {'0':  0, '1':  1, '2':  2, '3':  3, '4':  4, '5':  5, '6':  6, '7':  7, '8':  8, '9':  9,
          'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
          'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
          'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35,
          'a': 36, 'b': 37, 'c': 38, 'd': 39, 'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45,
          'k': 46, 'l': 47, 'm': 48, 'n': 49, 'o': 50, 'p': 51, 'q': 52, 'r': 53, 's': 54, 't': 55,
          'u': 56, 'v': 57, 'w': 58, 'x': 59, 'y': 60, 'z': 61}

inv_labels = {v: k for k, v in labels.items()}

db_names        = ["data/train_lmdb",   "data/val_lmdb"]
input_img_lists = ["data/train.csv",    "data/val.csv"]
input_img_dirs  = ["data/trainResized", "data/trainResized"]

# Data preparation
for db, input_list, input_dir in zip(db_names, input_img_lists, input_img_dirs):
    img_names, img_labels = tl.read_img_names_from_csv(input_list, skip_header=False, delimiter=",", append_string=".Bmp")

    img_idxs = [labels[cls] for cls in img_labels]

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
solver.step(100000)

# Testing data
net = caffe.Net(os.path.join(net_name, "deploy.prototxt"),
                os.path.join(net_name, "_iter_100000.caffemodel"),
                caffe.TEST)

transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
transformer.set_transpose("data", (2,0,1))
transformer.set_raw_scale("data", 255)
transformer.set_channel_swap("data", (2,1,0))

input_test_list = "data/test.csv"
img_test_names, img_test_labels = tl.read_img_names_from_csv(input_test_list, skip_header=False, delimiter=',')

test_dir = os.path.join("data", "testResized")

submission_file = "submission-" + time.strftime("%Y-%m-%d-%H-%M-%S") + ".csv"

with open(submission_file, 'wb') as csvfile:
    fieldnames = ["ID", "Class"]
    writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames)
    writer.writeheader()

    for id_ in img_test_names:
        net.blobs["data"].reshape(1, 3, 20, 20)
        img_path = os.path.join(test_dir, id_ + ".Bmp")
        net.blobs["data"].data[...] = transformer.preprocess("data", caffe.io.load_image(img_path))

        out = net.forward()
        class_ = inv_labels[out["prob"][0].argmax()]
        writer.writerow({"ID": id_, "Class": class_})
