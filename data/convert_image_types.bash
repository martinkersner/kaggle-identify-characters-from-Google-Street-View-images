#!/bin/bash

# Martin Kersner, m.kersner@gmail.com
# 2015/12/04

IMG_DIR="test/"
IMG_CONVERT_DIR="test_png/"
IMG_TYPE="png"

if ! [ -d "$IMG_CONVERT_DIR" ]; then
    mkdir "$IMG_CONVERT_DIR"
fi

for IMG_NAME in `ls $IMG_DIR`; do
    NEW_IMG_PATH="$IMG_CONVERT_DIR`echo $IMG_NAME | cut -d "." -f -1`.$IMG_TYPE"
    OLD_IMG_PATH="$IMG_DIR$IMG_NAME"

    convert "$OLD_IMG_PATH" -format $IMG_TYPE "$NEW_IMG_PATH"
done
