#!/usr/bin/python

# Martin Kersner, m.kersner@gmail.com
# 2015/11/23

# How to run:
# chmod +x train.py
# ./train settings_example.in

import os
import sys
import csv
import time
import numpy as np
import label_init as li
import tools as tl
tl.load_caffe()
import caffe
from pylab import *

# TODO 
# saving LOGS
# automaticaly save solver params
# save current model when Ctrl+C

def main():
    training_id = tl.current_time()
    tl.check_arguments(sys.argv, 1, "You have to specify settings file!\n./train.py settings_file")

    settings_filename = sys.argv[1]
    settings = tl.load_settings(settings_filename)
    prepare_data(settings)

    caffe.set_mode_gpu()
    solver = prepare_model(settings)
    train_loss, duration = train_model(solver, settings)

    plot_and_save_loss(train_loss, training_id, settings)
    print "\nDuration: " + sec2hms(duration)

def prepare_data(settings):
    db_names        = settings["db_names"]
    input_img_lists = settings["input_img_lists"]
    input_img_dirs  = settings["input_img_dirs"]

    for db, input_list, input_dir in zip(db_names, input_img_lists, input_img_dirs):
        img_names, img_labels = tl.read_img_names_from_csv(input_list, skip_header=False, delimiter=",", append_string=".png")
    
        img_idxs = [li.labels[cls] for cls in img_labels]
    
        if not tl.exist_dir(db):
            print "Creating " + db + " lmdb database."
            tl.imgs_to_lmdb(input_dir, img_names, db, labels=img_idxs)
        else:
            print "Database " + db + " already exists!"

def prepare_model(settings):
    model_path  = settings["model_path"] + ".prototxt"
    solver_path = settings["solver_path"] + ".caffemodel"
    
    solver = caffe.SGDSolver(solver_path)
    solver.net.copy_from(model_path) # TODO add condition for fine-tuning

    return solver

def train_model(solver, settings):
    n_iter = settings["n_iter"]
    train_loss = np.zeros(n_iter)

    start = time.clock()
    for it in range(n_iter):
        try:
            solver.step(1)
            train_loss[it] = solver.net.blobs['loss'].data
        except KeyboardInterrupt:
            end = time.clock()
            duration = end-start
            return train_loss, duration

    end = time.clock()
    duration = end-start

    return train_loss, duration

# TODO finish
def test_model(solver):
    test_iters = 10 # why 10?
    accuracy = 0

    for it in arange(test_iters):
        solver.test_nets[0].forward()
        accuracy += solver.test_nets[0].blobs['accuracy'].data
        accuracy /= test_iters

    return accuracy

def plot_and_save_loss(train_loss, training_id, settings):
    plot(np.vstack([train_loss]).T)
    plot_name = os.path.join(settings["model_dir"], "logs", "loss-" + training_id + ".png")
    savefig(plot_name, bbox_inches='tight')

def sec2hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    hms_str = "%02d:%02d:%02d" % (h, m, s)

    return hms_str

if __name__ == "__main__":
    main()
