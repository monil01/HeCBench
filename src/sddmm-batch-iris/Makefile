IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
ROCM_PATH ?= /opt/rocm
CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall
CPPFLAGS += -D__HIP_PLATFORM_AMD__ -I$(ROCM_PATH)/include -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64 -lhipsparse -ldl -lpthread
RUN_ARGS ?=
LAUNCHER ?=
all: main libhecbench_vendor.so
main: main.cpp utils.h ../../tools/iris_vendor_bridge.hpp Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp -o $@ $(LDFLAGS)
libhecbench_vendor.so: kernel.hip.cpp ../../tools/iris_vendor_bridge.cpp ../../tools/iris_vendor_bridge.hpp
	$(CXX) -std=c++17 -O2 -fPIC -shared -o $@ $<
run: all
ifneq ($(strip $(RUN_ARGS)),)
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
else
	IRIS_ARCHS=HIP $(LAUNCHER) ./main 32 1024 1024 1024 1024 1000 1
	IRIS_ARCHS=HIP $(LAUNCHER) ./main 32 4096 4096 4096 4096 1000 0
endif
clean:
	rm -f main libhecbench_vendor.so *.o
.PHONY: all run clean
