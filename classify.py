#!/usr/bin/python

# Martin Kersner, m.kersner@gmail.com
# 2015/11/26

import sys
import os
import csv
import label_init as li
import tools as tl
import numpy as np
tl.load_caffe()
import caffe

def main():
    tl.check_arguments(sys.argv, 1, "You have to specify settings file!\n./classify.py settings_file")
    settings_filename = sys.argv[1]
    settings = tl.load_settings(settings_filename)

    caffe_prototxt = settings["caffe_prototxt"]
    caffe_model    = settings["caffe_model"]
    test_list      = settings["test_list"]

    caffe.set_mode_gpu()
    net = caffe.Net(caffe_prototxt, caffe_model, caffe.TEST)
    
    transformer = create_transformer(net)
    img_test_names, img_test_labels = tl.read_img_names_from_csv(test_list, skip_header=False, delimiter=',')
    submission_file = classify_images(net, transformer, img_test_names, settings)

    print "Results written to " + submission_file

def create_transformer(net):
    transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
    transformer.set_transpose("data", (2,0,1))
    transformer.set_raw_scale("data", 255)
    transformer.set_channel_swap("data", (2,1,0))

    return transformer

def classify_images(net, transformer, img_test_names, settings):
    submission_file = "submission-" + tl.current_time() + ".csv"

    progress = 0
    progress_step = 100.0/len(img_test_names)

    with open(submission_file, 'wb') as csvfile:
        fieldnames = ["ID", "Class"]
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
    
        for id_ in img_test_names:
            progress += progress_step
            print str(np.round(progress, decimals=1)) + " %",
            sys.stdout.flush()
            print "\r",

            class_ = classify_image(net, transformer, id_, settings)
            writer.writerow({"ID": id_, "Class": class_})

    return submission_file

def classify_image(net, transformer, id_, settings):
    img_width  = settings["img_width"]
    img_height = settings["img_height"]
    test_dir   = settings["test_dir"]
    img_format = settings["img_format"]

    img_path = os.path.join(test_dir, id_ + "." + img_format)

    net.blobs["data"].reshape(1, 3, img_width, img_height)
    net.blobs["data"].data[...] = transformer.preprocess("data", caffe.io.load_image(img_path))

    out = net.forward()
    class_ = li.inv_labels[out["prob"][0].argmax()]

    return class_ 

if __name__ == "__main__":
    main()
