#!/bin/bash
git clone https://github.com/WongKinYiu/yolov9
cd yolov9
pip install -r requirements.txt

python train.py --img 640 --batch 4 --epochs 10 --data ../yolov9_config.yaml --weights yolov9.pt --device 0
