#!/usr/bin/python3
import os
import random
from csv import DictReader, DictWriter
from collections import OrderedDict
import sys
import math
import re
import argparse

DFT_GEOMETRY = "BLOCK"
CRASH_LOG = "panic.log"

class FaultDescriptor:
    def __init__(self, min_relative, max_relative, frame, layer, geometry_format):
        self.min_relative = min_relative 
        self.max_relative = max_relative
        self.frame = frame
        self.layer = layer # [0, YOLOv3CmdLine.conv_layer_num - 1]
        self.geometry_format = geometry_format # "LINE", "ALL", "BLOCK", "RANDOM", "SQUARE"
    
    def __str__(self):
        return (
            "Fault descriptor {\n"
            f"  Min relative: {self.min_relative}\n"
            f"  Max relative: {self.max_relative}\n"
            f"  Geometry format: {self.geometry_format}\n"
            f"  Frame: {self.frame}\n"
            f"  Layer: {self.layer}\n"
            "}"
        )
    
    __repr__ = __str__

    def writeToFile(self, f):
        f.write(f"{self.min_relative}\n")
        f.write(f"{self.max_relative}\n")
        f.write(f"{self.geometry_format}\n")
        f.write(f"{self.frame}\n")
        f.write(f"{self.layer}\n")
        f.write(f"\n")

class YOLOv3CmdLine:
    home_dir = "/home/rubens"
    # darknet_dir = f"{home_dir}/nvbitfi/test-apps/darknet_rlrj"
    darknet_dir = f"{home_dir}/darknet_rlrj"
    cfg_file = f"{darknet_dir}/cfg/yolov3-mix.cfg"
    weights_file = f"{darknet_dir}/yolov3.weights"
    data_file = f"{darknet_dir}/data/coco_frame_164_75x.txt"

    conv_layer_num = 75

    check_det_error_script = f"{darknet_dir}/check_detections_error.py"
    golden_stdout = f"{darknet_dir}/golden_stdout.txt"

    def createFaultDescriptorFile(self, fault_descriptors):
        tmp_fault_descriptor_file = f"/tmp/yolov3_rel_err_fault_descriptor.txt"
        with open(tmp_fault_descriptor_file, "w") as f:
            for fd in fault_descriptors:
                fd.writeToFile(f)
            f.close()
        return tmp_fault_descriptor_file

    def peformInjection(self, fault_descriptors, stdout_log):
        # Create fault descriptors file
        fault_descriptor_file = self.createFaultDescriptorFile(fault_descriptors)

        # Set environment variables
        export_vars = { "FAULT_PARAMETER_FILE": fault_descriptor_file }
        for export, val in export_vars.items():
            os.environ[export] = val

        # Execute shell command
        cmd = " ".join([
            f"{self.darknet_dir}/darknet detector rtest 10",
            self.cfg_file,
            self.weights_file,
            self.data_file,
            # Flags
            "-timedebug 1",
            # "-maxN 10",
        ])
        execute(cmd, stdout_log)

    def checkDetectionsError(self, inj_stdout, out_file):
        corruptedFrames = []

        # Execute Check Detections Errors script
        execute(f"python {self.check_det_error_script} {self.golden_stdout} {inj_stdout}", out_file)

        # Find corrupted frames
        regex = re.compile(r"Frame\s(\d+)")
        with open(out_file, "r") as dets_err_file:
            for line in dets_err_file:
                matches = regex.findall(line)
                if len(matches) > 0:
                    corruptedFrames.append(int(matches[0]))
            dets_err_file.close()
        
        return corruptedFrames

def execute(cmd, std_out_log="&1", log_info=None):
    print(f"EXECUTING: {cmd}")
    ret = os.system(cmd + f" >{std_out_log}")
    # Log any problem
    if ret != 0:
        with open(CRASH_LOG, 'a+') as fp:
            fp.write(f"Returning not zero: {ret}\nLog info: {log_info}\nCMD {cmd}\n")
    return ret

def injectAllLayersOneLayerPerFrame(relative_error, geometry_format, stdout_file):
    YOLO = YOLOv3CmdLine()

    # Create fault descriptors
    fault_descriptors = []
    for layer in range(YOLO.conv_layer_num):
        frame = layer # One layer per frame
        fd = FaultDescriptor(relative_error, relative_error, frame, layer, geometry_format)
        fault_descriptors.append(fd)

     # Perform relative error injection
    YOLO.peformInjection(fault_descriptors, stdout_file)


def perfomSingleInjection(relative_error, geometry_format=DFT_GEOMETRY,
                          stdout_file="/tmp/yolo_stdout.txt", dets_error_file="/tmp/det_errors_check_out.txt"):
    YOLO = YOLOv3CmdLine()
                        
    # Perform error injection
    injectAllLayersOneLayerPerFrame(relative_error, geometry_format, stdout_file)

    # Run detection errors check
    critCorruptedFrames = YOLO.checkDetectionsError(stdout_file, dets_error_file)

    return {
        "relative_error": relative_error,
        "stdout_file": stdout_file,
        "detections_error_file": dets_error_file,
        "conv_layers_that_caused_critical_sdcs": critCorruptedFrames, # one layer corrupted per frame
        "geometry_format": geometry_format,
    }

def performInjectionCampaign(out_log_file, geometry_format=DFT_GEOMETRY, skip_golden=False, N_layers=5):
    # Create/clean logs directory
    logs_path = "logs"
    execute(f"mkdir -p {logs_path}")
    execute(f"rm -f {logs_path}/*")
    execute(f"rm -f {CRASH_LOG}")

    if not skip_golden:
        # Generate golden stdout
        execute("make golden")

    layersRelErrThreshMap = OrderedDict()

    # Prepare output file
    with open(out_log_file, "w") as csv_output:
        header = ["it", "relative_error", "stdout_file", "detections_error_file", "conv_layers_that_caused_critical_sdcs", "geometry_format"]
        writer = DictWriter(csv_output, fieldnames=header, delimiter=";")
        writer.writeheader()

        relative_error_list = [x/100 for x in range(100, 1000+1, 1)]

        for (it, relative_error) in enumerate(relative_error_list):
            stdout_file = f"{logs_path}/stdout_rel_err_{relative_error}.txt"
            dets_error_file = f"{logs_path}/det_errors_{relative_error}.txt"

            inj_log = perfomSingleInjection(relative_error, stdout_file=stdout_file, dets_error_file=dets_error_file)
            
            for frame in critCorruptedFrames:
                layer = frame
                layersRelErrThreshMap[layer] = relative_error

            # Write injection log to file
            inj_log_line = { **inj_log, "it": it }
            writer.writerow(inj_log_line)
            print(inj_log_line)

            if len(layersRelErrThreshMap) >= N_layers:
                break
        
        csv_output.close()

    print("TERMINATING: ")
    print(layersRelErrThreshMap)

    if len(layersRelErrThreshMap) >= N_layers:
        exit(0)
    else:
        exit(1)
    
def binarySearchThresh(low=1.154156267, high=1.154156269, eps=1e-9):
    while (high - low) > eps:
        print("==================================================================")

        middle = (high - low)/2 + low
        inj_log = perfomSingleInjection(middle)
        print(inj_log)

        hasCriticalSDCs = len(inj_log["conv_layers_that_caused_critical_sdcs"]) > 0
        if hasCriticalSDCs:
            high = middle
        else:
            low = middle

        print(f"{low} - {high}")
        print("==================================================================\n")


def printUsage():
    print(f"Usage: {sys.argv[0]} <log-out-filename> ")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('function', action="store")
    parser.add_argument('-e', action="store", dest="relative_error", default=1.0)
    parser.add_argument('-o', action="store", dest="output_filename", default="rel_err_injection_log.csv")
    parser.add_argument('--geometry', action="store", dest="geometry_format", default=DFT_GEOMETRY)
    parser.add_argument('--skipGolden', action="store_true", dest="skip_golden", default=False,
            help=("Whether or not the Golden stdout file should be generated. "
                  "This flag can be set to false if the Golden stdout file was already generated previously."))

    args = parser.parse_args()

    if args.function == "campaign":
        performInjectionCampaign(args.output_filename, args.geometry_format, args.skip_golden)
    elif args.function == "single":
        inj_log_line = perfomSingleInjection(args.relative_error)
        print(inj_log_line)
    elif args.function == "binary-search":
        binarySearchThresh()

if __name__ == '__main__':
    main()
