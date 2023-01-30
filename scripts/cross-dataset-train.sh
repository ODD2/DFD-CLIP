#!/bin/bash

COMP=c23

accelerate launch main.py --cfg configs/cross-dataset/${COMP}/ff.yaml

