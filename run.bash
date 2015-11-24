#!/bin/bash

# Martin Kersner, m.kersner@gmail.com
# 2015/11/24

chmod +x split_train_val_list.bash
./split_train_val_list.bash

chmod +x train_and_classify.py
./train_and_classify.py
