IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
MPI_ROOT ?= /usr/lib/x86_64-linux-gnu/openmpi
ROCM_PATH ?= /opt/rocm
CXX ?= g++
HIPCC ?= hipcc
LAUNCHER ?= $(MPI_ROOT)/bin/mpirun -n 2
RUN_ARGS ?=
CXXFLAGS ?= -O3 -std=c++17 -Wall
CPPFLAGS += -D__HIP_PLATFORM_AMD__ -DOMPI_SKIP_MPICXX=1 -I$(MPI_ROOT)/include -I$(ROCM_PATH)/include -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(MPI_ROOT)/lib -Wl,-rpath,$(MPI_ROOT)/lib -lmpi -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lamdhip64 -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
all: main kernel.hip libhecbench_vendor.so
main: main.cpp ../../tools/iris_vendor_bridge.hpp Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) -O3 --genco -o $@ $<
libhecbench_vendor.so: vendor.cpp ../../tools/iris_vendor_bridge.cpp ../../tools/iris_vendor_bridge.hpp
	$(CXX) -O2 -std=c++17 -fPIC -shared -o $@ $<
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx libhecbench_vendor.so *.o
.PHONY: all run clean
