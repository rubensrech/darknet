GPU=0
CUDNN=0
OPENCV=0
OPENMP=0
DEBUG=0

ARCH= -gencode arch=compute_60,code=sm_60 \
      -gencode arch=compute_61,code=sm_61 \
      -gencode arch=compute_62,code=[sm_62,compute_62] \
      -gencode arch=compute_70,code=[sm_70,compute_70]
#      -gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?

# This is what I use, uncomment if you know your arch and want to specify
# ARCH= -gencode arch=compute_52,code=compute_52

VPATH=./src/:./examples
SLIB=libdarknet.so
ALIB=libdarknet.a
EXEC=darknet
OBJDIR=./obj/

CC=gcc
CPP=g++
NVCC=nvcc 
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON= -Iinclude/ -Isrc/
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC

ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` -lstdc++
COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN
LDFLAGS+= -lcudnn
endif

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o upsample_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o logistic_layer.o activation_layer.o rnn_layer.o gru_layer.o crnn_layer.o demo.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o l2norm_layer.o yolo_layer.o iseg_layer.o image_opencv.o
EXECOBJA=captcha.o lsd.o super.o art.o tag.o cifar.o go.o rnn.o segmenter.o regressor.o classifier.o coco.o yolo.o detector.o nightmare.o instance-segmenter.o darknet.o
ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o avgpool_layer_kernels.o
endif

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/darknet.h

all: obj backup results $(SLIB) $(ALIB) $(EXEC)
#all: obj  results $(SLIB) $(ALIB) $(EXEC)


$(EXEC): $(EXECOBJ) $(ALIB)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(OBJDIR)/*


test:
	gcc -c src/blas.c -I include/ -o tmp.o
	gcc -c src/option_list.c -I include/ -o tmp.o
	gcc -c src/im2col.c -I include/ -o tmp.o
	gcc -c src/gemm.c -I include/ -o tmp.o
	gcc -c src/activations.c -I include/ -o tmp.o
	gcc -c src/col2im.c -I include/ -o tmp.o
	gcc -c src/local_layer.c -I include/ -o tmp.o
	gcc -c src/utils.c -I include/ -o tmp.o
	gcc -c src/image.c -I include/ -o tmp.o
	gcc -c src/convolutional_layer.c -I include/ -o tmp.o
	gcc -c src/logistic_layer.c -I include/ -o tmp.o
	gcc -c src/deconvolutional_layer.c -I include/ -o tmp.o
	gcc -c src/l2norm_layer.c -I include/ -o tmp.o
	gcc -c src/activation_layer.c -I include/ -o tmp.o
	gcc -c src/rnn_layer.c -I include/ -o tmp.o
	gcc -c src/gru_layer.c -I include/ -o tmp.o
	gcc -c src/lstm_layer.c -I include/ -o tmp.o
	gcc -c src/crnn_layer.c -I include/ -o tmp.o
	gcc -c src/connected_layer.c -I include/ -o tmp.o
	gcc -c src/crop_layer.c -I include/ -o tmp.o
	gcc -c src/cost_layer.c -I include/ -o tmp.o
	gcc -c src/box.c -I include/ -o tmp.o
	gcc -c src/tree.c -I include/ -o tmp.o
	gcc -c src/region_layer.c -I include/ -o tmp.o
	gcc -c src/yolo_layer.c -I include/ -o tmp.o
	gcc -c src/iseg_layer.c -I include/ -o tmp.o
	gcc -c src/detection_layer.c -I include/ -o tmp.o
	gcc -c src/softmax_layer.c -I include/ -o tmp.o
	gcc -c src/normalization_layer.c -I include/ -o tmp.o
	gcc -c src/batchnorm_layer.c -I include/ -o tmp.o
	gcc -c src/maxpool_layer.c -I include/ -o tmp.o
	gcc -c src/reorg_layer.c -I include/ -o tmp.o
	gcc -c src/avgpool_layer.c -I include/ -o tmp.o
	gcc -c src/route_layer.c -I include/ -o tmp.o
	gcc -c src/upsample_layer.c -I include/ -o tmp.o
	gcc -c src/shortcut_layer.c -I include/ -o tmp.o
	gcc -c src/dropout_layer.c -I include/ -o tmp.o
	gcc -c src/data.c -I include/ -o tmp.o
	gcc -c src/network.c -I include/ -o tmp.o
	gcc -c src/matrix.c -I include/ -o tmp.o
	gcc -c src/parser.c -I include/ -o tmp.o
	gcc -c src/demo.c -I include/ -o tmp.o

	# Won't compile (even in original version)
	# examples/attention.c
	# examples/dice.c
	# examples/swag.c
	# examples/voxel.c
	# examples/writing.c

	gcc -c examples/art.c -I include/ -o tmp.o
	gcc -c examples/captcha.c -I include/ -o tmp.o
	gcc -c examples/cifar.c -I include/ -o tmp.o
	gcc -c examples/classifier.c -I include/ -o tmp.o
	gcc -c examples/coco.c -I include/ -o tmp.o
	gcc -c examples/detector.c -I include/ -o tmp.o
	gcc -c examples/go.c -I include/ -o tmp.o
	gcc -c examples/instance-segmenter.c -I include/ -o tmp.o
	gcc -c examples/lsd.c -I include/ -o tmp.o
	gcc -c examples/nightmare.c -I include/ -o tmp.o
	gcc -c examples/rnn_vid.c -I include/ -o tmp.o
	gcc -c examples/rnn.c -I include/ -o tmp.o
	gcc -c examples/segmenter.c -I include/ -o tmp.o
	gcc -c examples/super.c -I include/ -o tmp.o
	gcc -c examples/tag.c -I include/ -o tmp.o
	gcc -c examples/yolo.c -I include/ -o tmp.o
	gcc -c examples/darknet.c -I include/ -o tmp.o