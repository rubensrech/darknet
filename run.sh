#!/bin/bash
eval ${PRELOAD_FLAG} ${BIN_DIR}/darknet detector rtest 10 ${BIN_DIR}/cfg/yolov3.cfg ${BIN_DIR}/yolov3.weights ${BIN_DIR}/data/coco_100.txt > stdout.txt 2> stderr.txt