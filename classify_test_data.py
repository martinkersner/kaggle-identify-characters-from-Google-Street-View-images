#!/usr/bin/python

# Martin Kersner, m.kersner@gmail.com
# 2015/11/26

import os
import csv
import label_init as li
import tools as tl
tl.load_caffe()
import caffe

net_name = "bvlc_reference_caffenet"
W = 227
H = 227
img_format = "png"

# Testing data
net = caffe.Net(os.path.join(net_name, "deploy.prototxt"),
                os.path.join(net_name, "_iter_10000.caffemodel"),
                caffe.TEST)

print "Creating Transformer"
transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
transformer.set_transpose("data", (2,0,1))
transformer.set_raw_scale("data", 255)
transformer.set_channel_swap("data", (2,1,0))

print "Reading images"
input_test_list = "data/test.csv"
img_test_names, img_test_labels = tl.read_img_names_from_csv(input_test_list, skip_header=False, delimiter=',')

#test_dir = os.path.join("data", "testResized")
test_dir = os.path.join("data", "test_png_resized256")

submission_file = "submission-" + tl.current_time() + ".csv"

print "Creating submission file"
with open(submission_file, 'wb') as csvfile:
    fieldnames = ["ID", "Class"]
    writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames)
    writer.writeheader()

    for id_ in img_test_names:
        print id_
        net.blobs["data"].reshape(1, 3, W, H)
        img_path = os.path.join(test_dir, id_ + "." + img_format)
        net.blobs["data"].data[...] = transformer.preprocess("data", caffe.io.load_image(img_path))

        out = net.forward()
        class_ = li.inv_labels[out["prob"][0].argmax()]
        writer.writerow({"ID": id_, "Class": class_})
