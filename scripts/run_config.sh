#!/bin/bash
CFG_FILE=$1
NFRAMES=$2

OUTPUT_FILE="tests-results/results.txt"

# Run prediction on background
./darknet_rlrj/darknet detector rtest 2 $CFG_FILE yolov3.weights -n $NFRAMES >> $OUTPUT_FILE 2>> /dev/null &
PID=$!

# Calculate energy consumption
ENERGY=0
while ps -p $PID &>/dev/null; do
    CURR_POWER=$(cat /sys/bus/i2c/drivers/ina3221x/1-0040/iio_device/in_power0_input)
    ENERGY=$(($ENERGY + $CURR_POWER))
    sleep 1
done
ENERGY_PER_FRAME=$(($ENERGY / $NFRAMES))

echo "Total energy: $ENERGY mJ" >> $OUTPUT_FILE
echo "Energy per frame: $ENERGY_PER_FRAME mJ" >> $OUTPUT_FILE