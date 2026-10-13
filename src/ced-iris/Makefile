IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = g++
HIPCC ?= hipcc
RUN_ARGS ?= -a 0
LAUNCHER ?=
all: main kernel.hip
main: main.cpp kernel_reference.cpp kernel.h launch.h backend.h support/partitioner.h support/verify.h
	$(CXX) -O3 -std=c++17 -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp kernel_reference.cpp -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -lpthread -ldl
kernel.hip: kernel.hip.cpp kernels.h
	$(HIPCC) -O3 --genco kernel.hip.cpp -o $@
kernel.ptx: kernel.cu kernels.h
	nvcc -O3 -ptx kernel.cu -o $@
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
