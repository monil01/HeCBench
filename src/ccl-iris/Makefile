IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
ROCM_PATH ?= /opt/rocm
CXX ?= g++
MPI_ROOT ?= /usr/lib/x86_64-linux-gnu/openmpi
OPTIMIZE ?= yes
LAUNCHER ?= /usr/bin/mpirun -n 2
RUN_ARGS ?= 50
CXXFLAGS += -DOMPI_SKIP_MPICXX=1
CXXFLAGS += -std=c++17 -Wall -D__HIP_PLATFORM_AMD__ -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include -I$(ROCM_PATH)/include -I$(MPI_ROOT)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64 -lrccl -L$(MPI_ROOT)/lib -lmpi -ldl -lpthread
ifeq ($(OPTIMIZE),yes)
CXXFLAGS += -O3
endif
all: main libhecbench_vendor.so
main: main.cpp ../../tools/iris_vendor_bridge.hpp Makefile
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
libhecbench_vendor.so: kernel.hip.cpp ../../tools/iris_vendor_bridge.cpp ../../tools/iris_vendor_bridge.hpp
	$(CXX) -std=c++17 -O2 -fPIC -shared -o $@ $<
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main libhecbench_vendor.so *.o
.PHONY: all run clean
