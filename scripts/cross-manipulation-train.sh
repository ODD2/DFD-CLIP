#!/bin/bash

set -x
COMP=raw

accelerate launch main.py --cfg configs/cross-manipulation/${COMP}/df.yaml
accelerate launch main.py --cfg configs/cross-manipulation/${COMP}/f2f.yaml
accelerate launch main.py --cfg configs/cross-manipulation/${COMP}/fs.yaml
accelerate launch main.py --cfg configs/cross-manipulation/${COMP}/nt.yaml

