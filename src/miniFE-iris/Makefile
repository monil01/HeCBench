IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = g++
HIPCC ?= hipcc
TYPES = -DMINIFE_SCALAR=double -DMINIFE_LOCAL_ORDINAL=int -DMINIFE_GLOBAL_ORDINAL=int -DMINIFE_RESTRICT=__restrict__ -DMINIFE_CSR_MATRIX
CONFIG = -DMINIFE_INFO=0 -DMINIFE_KERNELS=0
INCLUDES = -Isrc -Iutils -Ifem
SOURCES = main.cpp src/YAML_Doc.cpp src/YAML_Element.cpp utils/BoxPartition.cpp utils/param_utils.cpp utils/utils.cpp utils/mytimer.cpp
HEADERS = $(wildcard src/*.hpp utils/*.hpp utils/*.h fem/*.hpp) launch.h backend.h
RUN_ARGS ?= -nx 128 -ny 128 -nz 128
LAUNCHER ?=
all: main kernel.hip
main: $(SOURCES) $(HEADERS)
	$(CXX) -O3 -std=c++17 $(TYPES) $(CONFIG) $(INCLUDES) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include $(SOURCES) -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -lpthread -ldl
kernel.hip: kernel.hip.cpp kernels.h
	$(HIPCC) -O3 $(TYPES) --genco kernel.hip.cpp -o $@
kernel.ptx: kernel.cu kernels.h
	nvcc -O3 $(TYPES) -ptx kernel.cu -o $@
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
