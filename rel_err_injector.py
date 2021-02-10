#!/usr/bin/python3
import os
import random
from csv import DictReader, DictWriter
import sys

CRASH_LOG = "panic.log"

class YOLOv3CMDLine:
    home_dir = "/home/rubens"
    # darknet_dir = f"{home_dir}/nvbitfi/test-apps/darknet_rlrj"
    darknet_dir = f"{home_dir}/darknet_rlrj"
    cfg_file = f"{darknet_dir}/cfg/yolov3.cfg"
    weights_file = f"{darknet_dir}/yolov3.weights"
    data_file = f"{darknet_dir}/data/coco_100.txt"

    cmd = " ".join([
        f"{darknet_dir}/darknet detector rtest 10",
        cfg_file,
        weights_file,
        data_file,
        # Flags
        "-timedebug 1",
        # "-maxN 1",
    ])

    conv_layer_num = 75
    csv_file = "rel_err_fault_model.csv"
    temporary_parameter_file = f"/tmp/yolov3_rel_err_fault_descriptor.txt"


def execute(cmd, std_out_log="&1", log_info=None):
    print(f"EXECUTING: {cmd}")
    ret = os.system(cmd + f" >{std_out_log}")
    # Log any problem
    if ret != 0:
        with open(CRASH_LOG, 'a+') as fp:
            fp.write(f"Returning not zero: {ret}\nLog info: {log_info}\nCMD {cmd}\n")


def run_yolov3_setup(std_out_log):
    export_vars = {
        "FAULT_PARAMETER_FILE": YOLOv3CMDLine.temporary_parameter_file,
    }

    for export, val in export_vars.items():
        os.environ[export] = val

    execute(YOLOv3CMDLine.cmd, std_out_log)


def main():
    # Create a tmp dir ./
    logs_path = "logs"
    execute(f"mkdir -p {logs_path}")
    execute(f"rm -f {logs_path}/*")
    execute(f"rm -f {CRASH_LOG}")

    parser_log = "rel_err_injection_log.csv"

    with open(YOLOv3CMDLine.csv_file, "r") as csv_input, open(parser_log, "w") as csv_output:
        reader = DictReader(csv_input)
        list_of_faults = list(reader)
        header = ["log_filename", 'min_relative', 'max_relative', 'geometry_format', "selected_layer", "it"]
        writer = DictWriter(csv_output, fieldnames=header)
        writer.writeheader()
        it = 0
        while list_of_faults:
            injection_data = random.choice(list_of_faults)
            necessary_info = ['min_relative', 'max_relative', 'geometry_format']
            with open(YOLOv3CMDLine.temporary_parameter_file, "w") as fp:
                # Write info related to flexgrip
                for info in necessary_info:
                    fp.write(f"{injection_data[info]}\n")

                # Write layer
                layer_random_choice = random.randint(0, YOLOv3CMDLine.conv_layer_num)
                fp.write(f"{layer_random_choice}\n")

            std_out_log_file = f"{logs_path}/{it}_stdout.txt"
            # Injection fault for line
            run_yolov3_setup(std_out_log_file)
            list_of_faults.remove(injection_data)

            line_dict = {
                **{"log_filename": std_out_log_file, "selected_layer": layer_random_choice, "it": it},
                **{k: v for k, v in injection_data.items() if k in necessary_info},
            }
            writer.writerow(line_dict)

            it += 1
            if it > 10: break


if __name__ == '__main__':
    main()
