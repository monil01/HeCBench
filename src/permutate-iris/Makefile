IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
HIPFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
RUN_ARGS ?= ../permutate-cuda/test_data/truerand_1bit.bin
LAUNCHER ?=
VALIDATION ?= 0
ifeq ($(VALIDATION),1)
CPPFLAGS += -DPERMUTATE_VALIDATION
endif
HOST = main.cpp permutation_testing.cpp utils.cpp statistical_test.cpp gpu_permutation_testing.cpp $(wildcard bzip/*.cpp)
HEADERS = header.h validation.h $(wildcard bzip/*.h)
all: main kernel.hip
main: $(HOST) $(HEADERS) Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $(HOST) $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp kernel_functions.cuh device_functions.cuh header.h Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ kernel.hip.cpp
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
