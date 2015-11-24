#!/bin/bash

# Martin Kersner, m.kersner@gmail.com
# 2015/11/23

train_ratio=0.5
data_dir="data/"
input_csv=$data_dir"trainLabels.csv"

train_file=$data_dir"train.csv"
val_file=$data_dir"val.csv"

if [ -f "$train_file" ] || [ -f "$val_file" ] ; then
    echo "Csv files with lists of images for training and validation already exist!"
    echo "Delete $train_file and $val_file if you want to generate lists again."
else
    tmp_csv="$input_csv"".tmp"
    
    sed '1d' "$input_csv" | sort -R > "$tmp_csv"
    num_samples=`wc -l $tmp_csv | cut -d " " -f 1 `
    
    num_train_samples=`echo "$num_samples"*"$train_ratio" | bc | cut -d "." -f 1`
    num_test_samples=`echo "$num_samples"-"$num_train_samples" | bc`
    
    head "$tmp_csv" -n "$num_train_samples" > $train_file
    tail "$tmp_csv" -n "$num_test_samples" > $val_file
    
    rm "$tmp_csv"
fi
