IRIS_ROOT ?= /tmp/iris
IRIS_LIBDIR ?= $(IRIS_ROOT)/build/src/runtime
CXX = g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
HIPFLAGS ?= -O3 -std=c++17
RUN_ARGS ?= ../opticalFlow-cuda/data/frame10.ppm ../opticalFlow-cuda/data/frame11.ppm
.PHONY: all run clean
all: main kernel.hip
main: main.cpp flowIRIS.cpp flowGold.cpp flowGold.h flowCUDA.h common.h dmem.hpp launchers.hpp
	$(CXX) $(CXXFLAGS) -ICommon -I$(IRIS_ROOT)/include main.cpp flowIRIS.cpp flowGold.cpp -o $@ -L$(IRIS_LIBDIR) -Wl,-rpath,$(IRIS_LIBDIR) -liris
kernel.hip: kernel.hip.cpp common.h tex2d_compat.h
	$(HIPCC) $(HIPFLAGS) --genco kernel.hip.cpp -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o FlowCPU.flo FlowGPU.flo
