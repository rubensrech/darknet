/*
 * parse_gemm_layer.cpp
 *
 *  Created on: Apr 11, 2020
 *      Author: fernando
 */

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <random>
#include <bits/stdc++.h>


#include "cuda_templates.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define DEBUG 0

typedef enum {
	SIMULATE_SCHEDULER_FAULT, INVALID_OPTION
} LayerOperationType;

LayerOperationType operation_type = SIMULATE_SCHEDULER_FAULT;

#define MAX_FLOAT_THRESHOLD 1.0e-5f

/**
 * Type that represents the fault description
 */
struct FaultDescriptor {
	float min_relative_error, max_relative_error;
	int frame_i, layer_i;
	std::string geometry_format;

	friend std::ostream& operator<<(std::ostream& os,
			const FaultDescriptor& fd) {
		os << "Min relative: " << fd.min_relative_error << std::endl;
		os << "Max relative: " << fd.max_relative_error << std::endl;
		os << "Frame: " << fd.frame_i << std::endl;
		os << "Layer: " << fd.layer_i << std::endl;
		os << "Geometry format: " << fd.geometry_format << std::endl;
		return os;
	}

	friend std::istream& operator>>(std::istream& is, FaultDescriptor& fd) {
		//Read the parameters from file
		is >> fd.min_relative_error;
		is >> fd.max_relative_error;
		is >> fd.geometry_format;
		is >> fd.frame_i;
		is >> fd.layer_i;
		return is;
	}
};

template<typename str_t>
std::string get_enviroment_var(str_t& src) {
	auto ptr = std::getenv(src);
	std::string ret = "";
	if (ptr) {
		ret = std::string(ptr);
	}
	return ret;
}

std::map<std::pair<int, int>, FaultDescriptor> layersFaultDescriptors;
bool faultParamFileRead = false;

void parse_fault_param_file() {
	std::string faultParamFilePath = get_enviroment_var("FAULT_PARAMETER_FILE");

	if (!faultParamFilePath.empty()) {
		if (DEBUG >= 1)
			std::cout << "Parsing FAULT_PARAMETER_FILE" << std::endl;

		std::ifstream faultParamFile(faultParamFilePath);

		if (faultParamFile.good()) {
			FaultDescriptor fd;
			while (faultParamFile >> fd) {
				layersFaultDescriptors[{ fd.frame_i, fd.layer_i }] = fd;
			}

			if (DEBUG >= 1)
				std::cout << layersFaultDescriptors.size() <<  " faults parsed in FAULT_PARAMETER_FILE" << std::endl;
			
			faultParamFile.close();
		} else {
			throw std::runtime_error("COULD NOT OPEN FILE: " + faultParamFilePath);
		}
	} else {
		if (DEBUG >= 1)
			std::cout << "FAULT_PARAMETER_FILE is not defined" << std::endl;
	}

	faultParamFileRead = true;
}

template<typename real_t>
void simulate_scheduler_fault(int M, int N, int frame_index, int conv_layer_index, std::vector<real_t>& C) {
	FaultDescriptor fd = layersFaultDescriptors.find({ frame_index, conv_layer_index })->second;
	
	std::random_device rd;
	std::mt19937 gen(rd());

	std::uniform_real_distribution<float> float_generator(fd.min_relative_error, fd.max_relative_error);
	std::uniform_int_distribution<int> bool_generator(0, 1);

	if (DEBUG >= 1) {
		std::cout << "DEBUG" << std::endl;
		std::cout << fd << std::endl;
	}

    if (fd.geometry_format == "BLOCK") {
		int x_start = std::max((M - BLOCK_SIZE)/2, 0), x_end = x_start + BLOCK_SIZE;
        int y_start = std::max((N - BLOCK_SIZE)/2, 0), y_end = y_start + BLOCK_SIZE;
        float rel_err = float_generator(gen);

		if (DEBUG >= 1) {
			std::cout << "Injecting into BLOCK " << BLOCK_SIZE << "x" << BLOCK_SIZE << std::endl;
			std::cout << "  Relative error: " << rel_err << std::endl;
			std::cout << "  Position: (" << x_start << "," << y_start << ")" << std::endl;
            std::cout << "  Matrix size: " << M << "x" << N << std::endl;
        }

		for (int i = x_start; i < x_end; i++) {
			for (int j = y_start; j < y_end; j++) {
				C[i * N + j] *= rel_err;
			}
		}
	} else {
        // Size selection
        std::uniform_int_distribution<int> int_m_generator(0, M - (BLOCK_SIZE < M ? BLOCK_SIZE : 0));
        std::uniform_int_distribution<int> int_p_generator(0, N - (BLOCK_SIZE < N ? BLOCK_SIZE : 0));
        auto start_i = int_m_generator(gen);
        auto start_j = int_p_generator(gen);

        if (fd.geometry_format == "RANDOM") {
            for (auto i = start_i; i < M; i++) {
                for (auto j = start_j; j < N; j++) {
                    auto is_necessary_to_inject = bool(
                            bool_generator(gen));
                    if (is_necessary_to_inject) {
                        C[i * N + j] *= float_generator(gen);
                    }
                }
            }
        } else if (fd.geometry_format == "SQUARE") {
            for (auto i = start_i; i < M; i++) {
                for (auto j = start_j; j < N; j++) {
                    C[i * N + j] *= float_generator(gen);
                }
            }
        } else if (fd.geometry_format == "ALL") {
            for (auto i = 0; i < M; i++) {
                for (auto j = 0; j < N; j++) {
                    C[i * N + j] *= float_generator(gen);
                }
            }
        } else if (fd.geometry_format == "LINE") {
            auto col_or_line = bool(bool_generator(gen));
            if (col_or_line) {
                // Select a line
                std::uniform_int_distribution<int> int_m_generator_line(0, M - 1);
                auto i = int_m_generator_line(gen);

                for (auto j = 0; j < N; j++) {
                    C[i * N + j] *= float_generator(gen);
                }
            } else {
                // Select a column
                std::uniform_int_distribution<int> int_n_generator_column(0, N - 1);
                auto j = int_n_generator_column(gen);
                for (auto i = 0; i < M; i++) {
                    C[i * N + j] *= float_generator(gen);
                }
            }
        }
    }
}

#ifdef GPU

template<typename real_t>
void parse_output_conv_layer_gpu(int TA, int TB, int M, int N, int K, real_t *C_gpu) {
	static int frame_index = 0;
	static int conv_layer_index = 0;

	if (!faultParamFileRead) {
		parse_fault_param_file();
	}

	if (DEBUG >= 1)
		std::cout << "Frame " << frame_index << ", Layer " << conv_layer_index << std::endl;

	if (layersFaultDescriptors.size() > 0) {
		if (layersFaultDescriptors.find({ frame_index, conv_layer_index }) != layersFaultDescriptors.end()) {
			/**
			 * If A is an m × n matrix and B is an n × p matrix,
			 * C is the m × p matrix
			 */
			auto size_c = M * N;

			std::vector<real_t> C_cpu(size_c);
			cuda_pull_array(C_gpu, C_cpu.data(), size_c);

			switch (operation_type) {
			case SIMULATE_SCHEDULER_FAULT: {
				if (DEBUG >= 3)
					std::cout << "Scheduler faults selected\n";
				simulate_scheduler_fault(M, N, frame_index, conv_layer_index, C_cpu);
				break;
			}

			case INVALID_OPTION:
				if (DEBUG >= 1)
					std::cout << "Invalid option\n";
				break;
			}

			cuda_push_array(C_gpu, C_cpu.data(), size_c);
		} else {
			if (DEBUG >= 2)
				std::cout << "No fault to be injected for layer " << conv_layer_index << " in frame " << frame_index << std::endl;
		}
	} else {
		if (DEBUG >= 2)
			std::cout << "No fault to be injected" << std::endl;
	}

	conv_layer_index++;
	if (conv_layer_index == 75) {
		frame_index++;
		conv_layer_index = 0;
	}
}

#endif

