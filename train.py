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
    check_cmd_arguments(sys.argv)

    settings_filename = sys.argv[1]
    settings = tl.load_settings(settings_filename)
    prepare_data(settings)

    set_computation_mode(settings)
    solver = prepare_model(settings)
    train_loss, duration = train_model(solver, settings)

    plot_and_save_loss(train_loss, training_id, settings)
    print "\nDuration: " + sec2hms(duration)

def check_cmd_arguments(argv):
    if (len(argv) != 2):
        print "You have to specify settings file!" 
        print "./train.py file"
        exit()

def prepare_data(settings):
    # caffe requires string type of str and not unicode
    db_names        = tl.unicode2str(settings["db_names"])
    input_img_lists = tl.unicode2str(settings["input_img_lists"])
    input_img_dirs  = tl.unicode2str(settings["input_img_dirs"])

    for db, input_list, input_dir in zip(db_names, input_img_lists, input_img_dirs):
        img_names, img_labels = tl.read_img_names_from_csv(input_list, skip_header=False, delimiter=",", append_string=".png")
    
        img_idxs = [li.labels[cls] for cls in img_labels]
    
        if not tl.exist_dir(db):
            print "Creating " + db + " lmdb database."
            tl.imgs_to_lmdb(input_dir, img_names, db, labels=img_idxs)
        else:
            print "Database " + db + " already exists!"

def set_computation_mode(settings):
    if (settings["comp_mode"].lower() == "gpu"):
        caffe.set_mode_gpu() 
    elif (settings["comp_mode"].lower() == "cpu"):
        caffe.set_mode_cpu()
    else:
        print "Computation mode was not correctly specified, GPU is used as default."
        caffe.set_mode_gpu() 

def prepare_model(settings):
    # caffe requires string type of str and not unicode
    model_dir   = str(settings["model_dir"])
    model_name  = str(settings["model_name"])
    solver_name = str(settings["solver_name"])
    
    solver_path = os.path.join(model_dir, solver_name + ".prototxt")
    model_path  = os.path.join(model_dir, model_name + ".caffemodel")
    
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
