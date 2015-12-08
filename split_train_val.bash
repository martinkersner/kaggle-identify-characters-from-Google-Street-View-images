#!/bin/bash

# Martin Kersner, m.kersner@gmail.com
# 2015/11/23

# Split train/val file to two separate files. 
# One used for training and one for validation.

# How to run:
# ./split_train_val.bash train_ratio train_val_file train_file val_file
# ./split_train_val.bash 0.5 data/trainLabels.csv train.csv val.csv

# It is expected that train_val_file contains header at the first line.

# Initialization
train_ratio=$1 # (0, 1)
train_val_path=$2
train_file=$3
val_file=$4

declare -a labels=('A' 'B' 'C' 'D' 'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L' 'M' \
                   'N' 'O' 'P' 'Q' 'R' 'S' 'T' 'U' 'V' 'W' 'X' 'Y' 'Z' \
                   'a' 'b' 'c' 'd' 'e' 'f' 'g' 'h' 'i' 'j' 'k' 'l' 'm' \
                   'n' 'o' 'p' 'q' 'r' 's' 't' 'u' 'v' 'w' 'x' 'y' 'z' \
                   '1' '2' '3' '4' '5' '6' '7' '8' '9' '0');

num_labels=${#labels[@]}

# Control parameters
below_zero=`echo "$train_ratio > 0" | bc`
upper_one=`echo "$train_ratio < 1" | bc`

if [[ $below_zero == 0 || $upper_one == 0 ]]; then
    echo "Training ratio has to be given within interval 0 and 1!"
    exit 1
fi

# Build paths to files
path=`dirname $train_val_path`
train_path=$path"/"$train_file
val_path=$path"/"$val_file

# Split training and validation data
if [ -f "$train_path" ] || [ -f "$val_path" ] ; then
    echo "Csv files with lists of images for training and validation already exist!"
    echo "Delete $train_path and $val_path if you want to generate lists again."
else
    touch $train_path
    touch $val_path

    tmp_csv="$train_val_path"".tmp"
    tmp_csv2="$train_val_path"".tmp2"

    sed '1d' "$train_val_path" | sort -R > "$tmp_csv"

    for (( c=0; c<=$num_labels-1; c++ )); do
        single_label=${labels[$c]} 
        cat "$tmp_csv" | grep ",$single_label" > "$tmp_csv2"

        num_label_samples=`wc -l "$tmp_csv2" | cut -d " " -f 1 `
        
        num_train_samples=`echo "$num_label_samples"*"$train_ratio" | bc | cut -d "." -f 1`
        num_test_samples=`echo "$num_label_samples"-"$num_train_samples" | bc`
        
        head "$tmp_csv2" -n "$num_train_samples" >> $train_path
        tail "$tmp_csv2" -n "$num_test_samples" >> $val_path
        
    done

    rm "$tmp_csv"
    rm "$tmp_csv2"
fi
