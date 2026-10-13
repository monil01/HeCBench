IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
RUN_ARGS ?= 1000
all: main kernel.hip
main: main.cpp sort.cpp dmem.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp sort.cpp -o $@ $(LDFLAGS)
kernel.hip: kernel.hip.cpp $(wildcard hip/*.cuh) hip/math_constants.h
	$(HIPCC) -O3 -DWARP_SIZE=32 -DHALF_WARP_SIZE=16 -I. --genco $< -o $@
kernel.ptx: kernel.cu $(wildcard cuda/*.cuh)
	nvcc -O3 -DWARP_SIZE=32 -DHALF_WARP_SIZE=16 -I. -ptx $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
