IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = g++
HIPCC ?= hipcc
RUN_ARGS ?= --niterations 100
LAUNCHER ?=
all: main kernel.hip
main: main.cpp utils_host.h utils.h memory.h launch.h backend.h
	$(CXX) -O3 -std=c++17 -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -lpthread -ldl
kernel.hip: kernel.hip.cpp kernels.h utils.h utils_device.h
	$(HIPCC) -O3 -munsafe-fp-atomics --genco kernel.hip.cpp -o $@
kernel.ptx: kernel.cu kernels.h utils.h utils_device.h
	nvcc -O3 -ptx kernel.cu -o $@
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
