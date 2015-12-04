# Identify characters from Google Street View images

This project tries to solve Kaggle competition (knowledge) called [First Steps With Julia](https://www.kaggle.com/c/street-view-getting-started-with-julia), however using only [Caffe framework](http://caffe.berkeleyvision.org/).

**Current score:** 0.69895

### TODO
* Add new networks
* Create cycle (create model, train, evaluate, submit result)
* Automatic submission

## Data description
Dataset consists of images depicting alphanumerical characters (lowercase [a-z], uppercase [A-Z] and [0-9]), therefore there are 62 different classes altogether.
All images differ in dimensions.
Given data are splitted into training a testing part.

## [Lenet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
Lenet is the first net we try to exploit for character identification.
Training is performed from scratch.

Maximal achieved score: 0.63711
