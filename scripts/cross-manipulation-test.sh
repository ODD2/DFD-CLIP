#!/bin/bash

declare -a arr=("DF" "F2F" "FS" "NT")
COMP=raw

set -x

for i in "${arr[@]}"
do
    echo "Start testing $i"
    accelerate launch inference.py logs/cross-manipulation-${COMP}/$i
done

