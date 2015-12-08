# Identify characters from Google Street View images

This project solves Kaggle competition (knowledge) called [First Steps With Julia](https://www.kaggle.com/c/street-view-getting-started-with-julia) using [Caffe framework](http://caffe.berkeleyvision.org/).

Martin Keršner, <m.kersner@gmail.com>

**Current top score:** 0.72284

#### TODO
* Add new networks
* Create cycle (create model, train, evaluate, submit result)
* Automatic submission
* Measure time of training
* Change a training script
* Save loss nad draw a plot at the end of training
* Run more instances of the same training algorithm at once (load parameters from external file json? csv?)

## Data description
Dataset (can be downloaded by [data/download_data.py](https://github.com/martinkersner/kaggle-identify-characters-from-Google-Street-View-images/blob/master/data/download_data.py)) consists of images depicting alphanumerical characters (lowercase [a-z], uppercase [A-Z] and [0-9]), so it includes 62 different classes altogether.
All images differ in dimensions, type of font and background and foreground color.
Given data are split into training (6,283 samples) a testing (6,220 samples) subsets. 
Dataset without augmentation will be denoted as *orig*.

<p align="center">
<img src="http://i.imgur.com/bAWbIpE.png?1" />
</p>

The data come originally from:

*T. E. de Campos, B. R. Babu and M. Varma, Character recognition in natural images, Proceedings of the International Conference on Computer Vision Theory and Applications (VISAPP), Lisbon, Portugal, February 2009.*

## Data augmentation
Most deep networks require static size of input, so the first step is to resize ([data/resize_images.py](https://github.com/martinkersner/kaggle-identify-characters-from-Google-Street-View-images/blob/master/data/resize_images.py)) images according to particular network. 


Distribution of classes within training subset is not weighted (shown in figure below). Some classes reach above 300 samples per class, whereas many of them don't even contain 60 samples per class.

<p align="center">
<img src="http://i.imgur.com/o9WJvaA.png?2" />
</p>

Number of training samples per class is insufficient for proper training and subsequent identification. 
One of the ways how to tackle this problem is to augment data and obtain overall higher number of training samples. 
In order to do that following methods can be employed:

* Rotation
* Applying filter

### Rotation
Many characters in dataset are slightly rotated and almost none of them are displayed straight.
Some of them are even upside down.
Therefore, we could anticipate that modest rotation of images could help to build a more robust model.

For a start, we rotate (about 10° using [data/rotate_dataset.m](https://github.com/martinkersner/kaggle-identify-characters-from-Google-Street-View-images/blob/master/data/rotate_dataset.m)) each of training images to the left and right once and create twice larger training dataset. It results in training dataset with 18,849 training images. This augmented dataset will be denoted as *rot_18849*.

<p align="center">
<img src="http://i.imgur.com/ltTwe3V.png?1" />
</p>

## Lenet
**Training from scratch**

[Lenet]((http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)) is the first net we try to exploit for character identification.

| Date | Data | Train/val rat. |Train. time | # iter. | Solver| Caffe ACC | Kaggle ACC | 
| :---:|:----:|:--------------:|:-------------:|:------------:|:-----:|:---------:|:----------:|
| 2015/11/26 | orig      | 50:50 | ?    | 10,000 | ? | ?    | 0.63711 |


## BVLC Caffenet
**Fine-tuning**

BVLC Caffenet is replication of [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) with few differences. This model was [trained](http://caffe.berkeleyvision.org/gathered/examples/imagenet.html) using [ImageNet dataset](http://www.image-net.org/). The model and more information can be found [here](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet).


| Date | Data | Train/val rat. |Train. time | # iter. | Solver| Caffe ACC | Kaggle ACC | 
| :---:|:----:|:--------------:|:-------------:|:------------:|:-----:|:---------:|:----------:|
| 2015/12/04 | orig      | 50:50 | ?    | 10,000 | ? | ?    | 0.69895 |
| 2015/12/07 | rot_18849 | 50:50 | 1h 58m | 10,000 | [params](https://github.com/martinkersner/kaggle-identify-characters-from-Google-Street-View-images/blob/master/bvlc_reference_caffenet/solver_params/2015_12_07_07_01.solverparams) | 0.944211 | **0.72284** |
