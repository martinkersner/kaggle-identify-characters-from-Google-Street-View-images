#!/bin/bash

# Martin Kersner, m.kersner@gmail.com
# 2015/11/23

# Split train/val file to two separate files. 
# One used for training and one for validation.

# How to run:
# ./split_train_val.bash train_ratio train_val.csv train.csv val.csv

# Initialization
train_ratio=$1 # (0, 1)
train_val_path=$2
train_file=$3
val_file=$4

# Control parameters
below_zero=`echo "$train_ratio > 0" | bc`
upper_one=`echo "$train_ratio < 1" | bc`

if [[ $below_zero == 0 || $upper_one == 0 ]]; then
    echo "Training ratio has to be given within interval 0 and 1!"
    exit 1
fi

path=`dirname $train_val_path`
train_path=$path"/"$train_file
val_path=$path"/"$val_file

if [ -f "$train_path" ] || [ -f "$val_path" ] ; then
    echo "Csv files with lists of images for training and validation already exist!"
    echo "Delete $train_path and $val_path if you want to generate lists again."
else
    tmp_csv="$train_val_path"".tmp"
    
    sed '1d' "$train_val_path" | sort -R > "$tmp_csv"
    num_samples=`wc -l $tmp_csv | cut -d " " -f 1 `
    
    num_train_samples=`echo "$num_samples"*"$train_ratio" | bc | cut -d "." -f 1`
    num_test_samples=`echo "$num_samples"-"$num_train_samples" | bc`
    
    head "$tmp_csv" -n "$num_train_samples" > $train_path
    tail "$tmp_csv" -n "$num_test_samples" > $val_path
    
    rm "$tmp_csv"
fi
