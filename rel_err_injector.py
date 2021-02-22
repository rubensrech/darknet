#!/usr/bin/python3
import os
import random
from csv import DictReader, DictWriter
import sys
import math

CRASH_LOG = "panic.log"

class YOLOv3CmdLine:
    home_dir = "/home/rubens"
    # darknet_dir = f"{home_dir}/nvbitfi/test-apps/darknet_rlrj"
    darknet_dir = f"{home_dir}/darknet_rlrj"
    cfg_file = f"{darknet_dir}/cfg/yolov3-mix.cfg"
    weights_file = f"{darknet_dir}/yolov3.weights"
    data_file = f"{darknet_dir}/data/coco_100.txt"

    conv_layer_num = 75

    check_det_error_script = f"{darknet_dir}/check_detections_error.py"
    golden_stdout = f"{darknet_dir}/golden_stdout.txt"

    def createFaultDescriptorFile(self, fault_descriptor):
        tmp_fault_descriptor_file = f"/tmp/yolov3_rel_err_fault_descriptor.txt"
        with open(tmp_fault_descriptor_file, "w") as fp:
            for info in fault_descriptor:
                fp.write(f"{fault_descriptor[info]}\n")
        fp.close()
        return tmp_fault_descriptor_file

    def peformInjection(self, injection_data, stdout_log):
        # Create fault_descriptor
        necessary_info = ["min_relative", "max_relative", "geometry_format"]
        fault_descriptor = {}
        for info in necessary_info: fault_descriptor[info] = injection_data[info]
        fault_descriptor["layer"] = injection_data["layer"] if "layer" in injection_data else random.randint(0, self.conv_layer_num)
        fault_descriptor_file = self.createFaultDescriptorFile(fault_descriptor)

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
            "-maxN 10",
        ])
        execute(cmd, stdout_log)

        return { "stdout_file": stdout_log, **fault_descriptor }

    def checkDetectionsError(self, inj_stdout, out_file):
        execute(f"python {self.check_det_error_script} {self.golden_stdout} {inj_stdout}", out_file)

def execute(cmd, std_out_log="&1", log_info=None):
    print(f"EXECUTING: {cmd}")
    ret = os.system(cmd + f" >{std_out_log}")
    # Log any problem
    if ret != 0:
        with open(CRASH_LOG, 'a+') as fp:
            fp.write(f"Returning not zero: {ret}\nLog info: {log_info}\nCMD {cmd}\n")
    return ret

def performInjectionCampaign(faults_csv_file, injection_order="ORDERED"):
    # Create/clean logs directory
    logs_path = "logs"
    parser_log = "rel_err_injection_log.csv"
    execute(f"mkdir -p {logs_path}")
    execute(f"rm -f {logs_path}/*")
    execute(f"rm -f {CRASH_LOG}")

    # Generate golden stdout
    execute("make clean")
    execute("make INJECT_REL_ERROR=0")
    execute("make golden")

    # Recompile for Relative Error injection
    execute("make clean")
    execute("make INJECT_REL_ERROR=1")

    with open(faults_csv_file, "r") as faults_csv_input, open(parser_log, "w") as csv_output:
        # Parse faults file
        reader = DictReader(faults_csv_input)
        list_of_faults = list(reader)

        # Prepare output file
        header = ["it", 'min_relative', 'max_relative', 'geometry_format', "layer", "stdout_file", "det_err_file", "detection_corrupted"]
        writer = DictWriter(csv_output, fieldnames=header)
        writer.writeheader()

        it = 0
        YOLO = YOLOv3CmdLine()
        while list_of_faults:
            # Perform relative error injection
            injection_data = random.choice(list_of_faults) if injection_order == "RANDOM" else list_of_faults[0]
            stdout_log_file = f"{logs_path}/{it}_stdout.txt"
            injection_log = YOLO.peformInjection(injection_data, stdout_log_file)

            # Run detection errors check
            det_error_out_file = f"{logs_path}/{it}_det_errors.txt"
            YOLO.checkDetectionsError(stdout_log_file, det_error_out_file)
            detection_corrupted = os.stat(det_error_out_file).st_size > 0

            # Write injection log to file
            writer.writerow({
                "it": it,
                "stdout_file": stdout_log_file,
                "det_err_file": det_error_out_file,
                "detection_corrupted": detection_corrupted,
                **injection_log,
            })

            list_of_faults.remove(injection_data)
            it += 1

def generateInjectionsFile(out_filename, mode="LINEAR"):
    with open(out_filename, "w") as f:
        header = ["layer", "min_relative", "max_relative", "geometry_format"]
        writer = DictWriter(f, fieldnames=header)
        writer.writeheader()

        if mode == "LINEAR":
            layer = 5 # [0, YOLOv3CmdLine.conv_layer_num - 1]
            min   = 1.1
            max   = 2.0
            step  = 0.1
            geometry = "BLOCK"
            n = math.ceil((max - min) / step)+1
            
            for i in range(n):
                rel_err = min + i*step
                writer.writerow({
                    "layer": layer,
                    "min_relative": rel_err,
                    "max_relative": rel_err,
                    "geometry_format": geometry
                })

            print(f"LINEAR: {n} faults generated")
            print(f"    - layer: {layer}")
            print(f"    - relative error range: [{min},{max}]")
            print(f"    - geometry: {geometry}")

        f.close()

def printUsage(function=None):
    if function == None:
        print(f"Usage: {sys.argv[0]} <inject|gen-injs-file> <args>")
    else:
        if function == "inject": 
            print(f"Usage: {sys.argv[0]} inject <faults-csv-file> [<injection-order=ORDERED|RANDOM>]")
        elif function == "gen-injs-file":
            print(f"Usage: {sys.argv[0]} gen-injs-file <out-filename> [<mode=LINEAR>]")

def main():
    if len(sys.argv) >= 2:
        function = sys.argv[1]

        if function == "inject":
            if len(sys.argv) >= 3:
                faults_csv_file = sys.argv[2]
                injection_order = sys.argv[3] if len(sys.argv) >= 4 else "ORDERED"
                performInjectionCampaign(faults_csv_file, injection_order)
            else:
                printUsage(function)

        elif function == "gen-injs-file":
            if len(sys.argv) >= 3:
                out_filename = sys.argv[2]
                mode = sys.argv[3] if len(sys.argv) >= 4 else "LINEAR"
                generateInjectionsFile(out_filename, mode)
            else:
                printUsage(function)

        else:
            printUsage(function=None)
    else:
        printUsage()
        

if __name__ == '__main__':
    main()
