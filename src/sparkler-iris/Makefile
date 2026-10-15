IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
HIPCC ?= hipcc
MPICXX ?= /usr/bin/mpicxx
MPIEXEC ?= /usr/bin/mpirun
HIP_ARCH ?= native
CXXFLAGS ?= -O3 -std=c++17
MPI_CFLAGS := $(shell $(MPICXX) --showme:compile)
MPI_LIBS := $(shell $(MPICXX) --showme:link)
RUN_ARGS ?= --num_vector 4000 --num_field 90000 --num_iterations 10
RANKS ?= 4
LAUNCHER ?=
all: main kernel.hip libhecbench_vendor.so
main: main.cpp observe.hpp Makefile ../../tools/iris_vendor_bridge.hpp
	$(HIPCC) $(CXXFLAGS) $(MPI_CFLAGS) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include $< -o $@ $(MPI_LIBS) -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -lhipblas -ldl -lpthread
kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) -O3 --offload-arch=$(HIP_ARCH) --genco $< -o $@
libhecbench_vendor.so: bridge.cpp ../../tools/iris_vendor_bridge.cpp ../../tools/iris_vendor_bridge.hpp
	$(HIPCC) -O2 -std=c++17 -fPIC -shared $< -o $@
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) $(MPIEXEC) -n $(RANKS) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip libhecbench_vendor.so *.o
