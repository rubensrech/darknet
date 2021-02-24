#!/bin/bash

NFRAMES=10

ERROR_CFGS_FILE="error-cfgs.txt"
PERF_CFGS_FILE="performance-cfgs.txt"
COST_BEN_CFGS_FILE="cost-benefit-cfgs.txt"

[ ! -f "$ERROR_CFGS_FILE" ] && echo "$ERROR_CFGS_FILE does not exist"
[ ! -f "$PERF_CFGS_FILE" ] && echo "$PERF_CFGS_FILE does not exist"
[ ! -f "$COST_BEN_CFGS_FILE" ] && echo "$COST_BEN_CFGS_FILE does not exist"

ERROR_RESULTS_FILE="tests-results/results-error-cfgs.txt"
PERF_RESULTS_FILE="tests-results/results-performance-cfgs.txt"
COST_BEN_RESULTS_FILE="tests-results/results-cost-benefit-cfgs.txt"

[ -f "$ERROR_RESULTS_FILE" ] && echo "$ERROR_RESULTS_FILE already exists!" && exit -1
[ -f "$PERF_RESULTS_FILE" ] && echo "$PERF_RESULTS_FILE already exists!" && exit -1
[ -f "$COST_BEN_RESULTS_FILE" ] && echo "$COST_BEN_RESULTS_FILE already exists!" && exit -1

for currCfgFile in "$ERROR_CFGS_FILE" "$PERF_CFGS_FILE" "$COST_BEN_CFGS_FILE"
do
    echo "PROCESSING $currCfgFile"
    RESULTS_FILE_NAME="results-$currCfgFile"
    RESULTS_FILE="tests-results/$RESULTS_FILE_NAME"

    while IFS= read -r line
    do
        # Set mixed-precision config
        cd darknet_rlrj
        printf "2\n$line\n" | python3 scripts/setLayersPrecision.py cfg/yolov3-mix.cfg
        cd ..

        NHALF_LAYERS=$(($(grep -o ',' <<< "$line" | grep -c .) + 1))
        echo " - CONFIG: $NHALF_LAYERS FP16 Layers"
        
        echo "=================================" >> $RESULTS_FILE
        echo "CONFIG: $NHALF_LAYERS FP16 Layers" >> $RESULTS_FILE
        
        # Run prediction twice
        ./run_single_config yolov3-mix.cfg $NFRAMES $RESULTS_FILE_NAME
        ./run_single_config yolov3-mix.cfg $NFRAMES $RESULTS_FILE_NAME

        printf "\n\n" >> $RESULTS_FILE

    done < "$currCfgFile"
done