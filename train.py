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
import io
import numpy as np
import label_init as li
import tools as tl
tl.load_caffe()
import caffe
from pylab import *

# TODO saving CAFFE LOGS

def main():
    tl.check_arguments(sys.argv, 1, "You have to specify settings file!\n./train.py settings_file")

    settings_filename = sys.argv[1]
    settings = tl.load_settings(settings_filename)

    train_id = tl.generate_unique_id()
    settings = modify_solver(train_id, settings)

    caffe_stream = io.StringIO()
    with tl.redirect_caffe_log(caffe_stream): 
        caffe.set_mode_gpu()
        prepare_data(settings)
        solver = prepare_model(settings)
        train_loss, duration, iteration = train_model(solver, settings)

    log(solver, train_loss, duration, iteration, caffe_stream, settings)

    #with open("help.txt", 'wb') as f2:
    #    f2.write(caffe_stream.getvalue().decode('utf-8'))

def modify_solver(train_id, settings):
    log_dir = tl.create_log_dir(settings["log_dir"], train_id)
    new_solver_path = os.path.join(log_dir, "solver.prototxt")

    tl.modify_solver_parameter(settings["solver_path"], 
                               new_solver_path,
                               "snapshot_prefix", 
                               os.path.join(settings["log_dir"], train_id) + "/")

    settings["solver_path"] = new_solver_path
    settings["log_dir"]     = log_dir
    
    return settings

def log(solver, train_loss, duration_int, iteration, caffe_stream, settings):
    log_dir = settings["log_dir"]
    tl.log_file(settings["settings_filename"], log_dir)
    tl.log_file(settings["input_img_lists"][0], log_dir)
    tl.log_file(settings["input_img_lists"][1], log_dir)

    # loss
    log_loss(train_loss, log_dir)

    # duration
    duration_str = tl.log_to_file(log_dir, "duration", tl.sec2hms(duration_int))

    # save model
    solver.net.save(os.path.join(log_dir, "final.caffemodel"))

    # terminated training
    if (iteration != None):
        # save current iteration
        tl.log_to_file(log_dir, "termination", iteration)
        print "Terminated: " + str(iteration) + " iterations"

    # caffe log
    caffe_log_path = os.path.join(settings["log_dir"], "caffe.log")
    with open(caffe_log_path, 'wb') as f:
        f.write(caffe_stream.getvalue().decode('utf-8'))

    print "Duration: " + duration_str
    print "Logged to " + log_dir

def log_loss(train_loss, log_dir):
    plot(np.vstack([train_loss]).T)
    plot_name = os.path.join(log_dir, "loss.png")
    savefig(plot_name, bbox_inches='tight')
    
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
    solver_path = settings["solver_path"]
    model_path  = settings["model_path"]
    
    solver = caffe.SGDSolver(solver_path)
    solver.net.copy_from(model_path) # TODO add condition for training from scratch

    return solver

def train_model(solver, settings):
    n_iter = settings["n_iter"]
    train_loss = np.zeros(n_iter)
    terminated_it = None

    start = time.clock()
    for it in range(n_iter):
        try:
            solver.step(1)
            train_loss[it] = solver.net.blobs['loss'].data
        except KeyboardInterrupt:
            end = time.clock()
            duration = end-start
            terminated_it = it
            return train_loss, duration, terminated_it

    end = time.clock()
    duration = end-start

    return train_loss, duration, terminated_it

# TODO DEPRECATED?
#def test_model(solver, settings):
#    test_iter = 10 # why 10?
#    accuracy = 0
#
#    for it in arange(test_iter):
#        solver.test_nets[0].forward()
#        accuracy += solver.test_nets[0].blobs['accuracy'].data
#        accuracy /= test_iters
#
#    return accuracy

if __name__ == "__main__":
    main()
