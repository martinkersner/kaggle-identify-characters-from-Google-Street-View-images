#!/usr/bin/python

# Martin Kersner, m.kersner@gmail.com
# 2015/11/26

import sys
import os
import csv
import label_init as li
import tools as tl
from ProgressBar import *
from ConfusionMatrix import *
import numpy as np
tl.load_caffe()
import caffe

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from collections import Counter

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

    img_test_names, img_test_labels = tl.read_img_names_from_csv(test_list,
                                                                 skip_header=False,
                                                                 delimiter=',')

    submission_file, confusion_matrix, accuracy = classify_images(net,
                                                                  transformer,
                                                                  img_test_names,
                                                                  img_test_labels,
                                                                  settings)

    if (settings["print_csv"]):
        print "Results written to " + submission_file

    print "Accuracy: " + str(accuracy)

    # Confusion matrix
    if (settings["conf_matrix"]):
        plt.figure(figsize=(2,2), dpi=200)
        plt.clf()
        plt.imshow(confusion_matrix, norm=LogNorm())
        #plt.colorbar()
        plt.savefig(settings["conf_matrix_path"])

def create_transformer(net):
    transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
    transformer.set_transpose("data", (2,0,1))
    transformer.set_raw_scale("data", 255)
    transformer.set_channel_swap("data", (2,1,0))

    return transformer

def classify_images(net, transformer, img_test_names, img_test_labels, settings):
    pb = ProgressBar(len(img_test_names))
    cm = ConfusionMatrix(li.labels)

    pred_hit = np.zeros(len(img_test_names))
    class_preds = []

    for i, (id_, true_class) in enumerate(zip(img_test_names, img_test_labels)):
        pb.print_progress()

        if (settings["bulk"]):
            settings["test_dir"] = settings["bulk_dir"]
            pred_class = classify_from_bulk(net, transformer, id_, true_class, settings)
        else:
            pred_class = classify_image(net, transformer, id_, settings)[0]

        class_preds.append(pred_class)

        if (pred_class == true_class):
            pred_hit[i] = 1

        cm.actualize(true_class, pred_class)

    submission_file = print_submission(img_test_names, class_preds, settings)

    accuracy = compute_accuracy(pred_hit)

    return submission_file, cm.get_confusion_matrix(), accuracy

def print_submission(img_test_names, class_preds, settings):
    submission_file = None

    if (settings["print_csv"]):
        submission_file = print_csv(img_test_names, class_preds, settings)
    elif (settings["print_term"]):
        print_term(img_test_names, class_preds, settings)

    return submission_file

def print_csv(img_ids, class_preds, settings):
    field_names = settings["field_names"]
    fn_id    = field_names[0] 
    fn_class = field_names[1] 

    submission_file = "submission-" + tl.current_time() + ".csv"

    with open(submission_file, 'wb') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=field_names)
        writer.writeheader()

        for id_, pred in zip(img_ids, class_preds):
            writer.writerow({fn_id: id_, fn_class: pred})

    return submission_file

def print_term(img_ids, class_preds, settings):
    fn_id    = settings["field_names"][0]
    fn_class = settings["field_names"][1]

    print fn_id + "," + fn_class 

    for id_, pred in zip(img_ids, class_preds):
        print id_ + "," + pred

def compute_accuracy(pred_hit):
    return (1.0*np.sum(pred_hit))/len(pred_hit)

def classify_from_bulk(net, transformer, id_, true_class, settings):
    bulk_suffixes = settings["bulk_suffixes"]
    bulk_size     = settings["bulk_size"]

    pred_classes = []

    for suffix in bulk_suffixes:
        img_name = create_bulk_name([id_, true_class, suffix], '-')
        pred_classes.extend(classify_image(net, transformer, img_name, settings)[0:bulk_size])

    class_counter = Counter(pred_classes)
    return most_frequent(class_counter)

def create_bulk_name(names, delimiter):
    bulk_name = ""
    last_name = names.pop()

    for name in names:
        bulk_name += name + delimiter

    bulk_name += last_name

    return bulk_name

def most_frequent(class_counter):
    return class_counter.most_common(1)[0][0]

def classify_image(net, transformer, id_, settings):
    img_width  = settings["img_width"]
    img_height = settings["img_height"]
    test_dir   = settings["test_dir"]
    img_format = settings["img_format"]

    img_path = os.path.join(test_dir, id_ + "." + img_format)

    net.blobs["data"].reshape(1, 3, img_width, img_height) # does it have to be executed for each image?
    net.blobs["data"].data[...] = transformer.preprocess("data", caffe.io.load_image(img_path))

    out = net.forward()

    # return all predicted classes sorted according to prob
    sort_idxs = out["prob"][0].argsort()[::-1]
    classes = [li.inv_labels[cls] for cls in sort_idxs]

    return classes

if __name__ == "__main__":
    main()
