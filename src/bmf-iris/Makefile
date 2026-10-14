IRIS_ROOT ?= /tmp/iris
HIP_ROOT ?= /opt/rocm
CXX ?= g++
HIPCC ?= $(HIP_ROOT)/bin/hipcc
RUN_ARGS ?= data/MNIST.in -r 8 -l 112 -i 10000 --seed 123 --stuck 10000
CXXFLAGS ?= -O3 -std=c++17 -fopenmp
all: main kernel.hip
main: main.cpp $(wildcard *.h *.hpp *.cuh helper/*)
	$(CXX) $(CXXFLAGS) -DUSE_GPU -DWARPSPERBLOCK=8 -D__HIP_PLATFORM_AMD__ -I$(HIP_ROOT)/include -I$(IRIS_ROOT)/include $< -o $@ -L$(IRIS_ROOT)/build/src/runtime -Wl,-rpath,$(IRIS_ROOT)/build/src/runtime -liris -L$(HIP_ROOT)/lib -Wl,-rpath,$(HIP_ROOT)/lib -lamdhip64
kernel.hip: kernel.hip.cpp $(wildcard *.h *.hpp *.cuh helper/*)
	$(HIPCC) -O3 -DWARPSPERBLOCK=8 --genco $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip
