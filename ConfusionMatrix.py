#!/usr/bin/python

# Martin Kersner, m.kersner@gmail.com
# 2015/12/10

import numpy as np

class ConfusionMatrix:
    def __init__(self, labels):
        self.labels= labels
        num_labels = len(self.labels)
        self.confusion_matrix = np.zeros((num_labels, num_labels), dtype=np.uint)

    def actualize(self, true_class, pred_class):
        self.confusion_matrix[self.labels[true_class], self.labels[pred_class]] += 1

    def get_confusion_matrix(self):
        return self.confusion_matrix
