IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
RUN_ARGS ?= mcmd.inp
DATA_DIR ?= ../mcmd-cuda/dataset
CXXFLAGS ?= -O3 -std=c++17
all: main kernel.hip
main: main.cpp Makefile
	$(CXX) $(CXXFLAGS) -DGPU -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread -o $@
kernel.hip: kernel.hip.cpp
	$(HIPCC) -O3 --genco kernel.hip.cpp -o $@
run: all
	cd $(DATA_DIR) && IRIS_ARCHS=HIP IRIS_KERNEL_BIN_HIP=$(abspath kernel.hip) $(abspath main) $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
