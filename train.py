#!/usr/bin/python

# Martin Kersner, m.kersner@gmail.com
# 2015/11/23

import os
import sys
import csv
import json
import numpy as np
import label_init as li
import tools as tl
tl.load_caffe()
import caffe

# TODO saving LOGS

def main():
    if (len(sys.argv) != 2):
        print "You have to specify settings file!" 
        print "./train.py file"
        exit()

    caffe.set_mode_gpu() # TODO add to settings file
    settings = load_settings(sys.argv[1])
    prepare_data(settings)
    solver = prepare_model(settings)
    
    # Training
    n_iter = settings["n_iter"]
    train_loss = np.zeros(n_iter)
    for it in range(n_iter):
        solver.step(1)
        train_loss[it] = solver.net.blobs['loss'].data

def load_settings(settings_filename):
    try:
        with open(settings_filename) as settings_file:    
            settings = json.load(settings_file)
    except IOError as e:
        print "Unable to open settings file!"
        exit()

    return settings

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
    # caffe requires string type of str and not unicode
    model_dir   = str(settings["model_dir"])
    model_name  = str(settings["model_name"])
    solver_name = str(settings["solver_name"])
    
    solver_path = os.path.join(model_dir, solver_name + ".prototxt")
    model_path  = os.path.join(model_dir, model_name + ".caffemodel")
    
    solver = caffe.SGDSolver(solver_path)
    solver.net.copy_from(model_path) # TODO add condition for fine-tuning

    return solver

if __name__ == "__main__":
    main()
