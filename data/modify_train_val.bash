#!/bin/bash

# Martin Kersner, m.kersner@gmail.com
# 2015/12/08

old_file="train.csv"
new_file="train_expanded.csv"
new_file_tmp="$new_file"".tmp"

touch "$new_file_tmp"

while IFS='' read -r line || [[ -n "$line" ]]; do
    label=`echo $line | cut -d "," -f 2 | tr -d '\r\n'`

    echo $line | sed "s@,@-$label-or,@"  >> "$new_file_tmp"

    echo $line | sed "s@,@-$label-l1,@"  >> "$new_file_tmp"
    echo $line | sed "s@,@-$label-l2,@"  >> "$new_file_tmp"
    echo $line | sed "s@,@-$label-l3,@"  >> "$new_file_tmp"
    echo $line | sed "s@,@-$label-l4,@"  >> "$new_file_tmp"

    echo $line | sed "s@,@-$label-r1,@" >> "$new_file_tmp"
    echo $line | sed "s@,@-$label-r2,@" >> "$new_file_tmp"
    echo $line | sed "s@,@-$label-r3,@" >> "$new_file_tmp"
    echo $line | sed "s@,@-$label-r4,@" >> "$new_file_tmp"
done < $old_file

cat "$new_file_tmp" | sort -R > "$new_file"
rm "$new_file_tmp"
