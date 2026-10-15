IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
ROCM_PATH ?= /opt/rocm
CXX ?= g++
HIPCC ?= hipcc
RUN_ARGS ?= 100
CXXFLAGS ?= -O3 -std=c++17
all: main kernel.hip libhecbench_vendor.so
main: main.cpp format.h ../../tools/iris_vendor_bridge.hpp
	$(CXX) $(CXXFLAGS) -D__HIP_PLATFORM_AMD__ -I$(ROCM_PATH)/include -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include $< -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lhipblas -lamdhip64 -ldl -lpthread -o $@
kernel.hip: kernel.hip.cpp format.h
	$(HIPCC) -O3 --genco $< -o $@
libhecbench_vendor.so: vendor.cpp ../../tools/iris_vendor_bridge.cpp ../../tools/iris_vendor_bridge.hpp
	$(CXX) -O2 -std=c++17 -fPIC -shared $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip libhecbench_vendor.so *.o
.PHONY: all run clean
