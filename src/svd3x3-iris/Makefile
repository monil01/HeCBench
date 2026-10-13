IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = g++
HIPCC ?= hipcc
RUN_ARGS ?= ../svd3x3-cuda/Dataset_1M.txt 1000
LAUNCHER ?=
all: main kernel.hip
main: main.cpp svd_host.h backend.h
	$(CXX) -O3 -ffast-math -std=c++17 -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -lpthread -ldl
kernel.hip: kernel.hip.cpp kernels.h
	$(HIPCC) -O3 -ffast-math --genco kernel.hip.cpp -o $@
kernel.ptx: kernel.cu kernels.h
	nvcc -O3 -use_fast_math -ptx kernel.cu -o $@
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
