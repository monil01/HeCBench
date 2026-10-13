IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
NVCC ?= nvcc
RUN_ARGS ?= -i 1000 -l 32 -t 128 -v 3 -w 1
LAUNCHER ?=
CXXFLAGS ?= -O3 -std=c++17 -Wall
PRECISION ?= 1
CPPFLAGS += -DMILC_COMPLEX -DPRECISION=$(PRECISION)
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
.PHONY: all run clean
all: main kernel.hip
main: main.cpp iris_support.h lattice.hpp su3.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp -o $@ $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp lattice.hpp su3.h Makefile
	$(HIPCC) -DMILC_COMPLEX -DPRECISION=$(PRECISION) -O3 --genco -o $@ kernel.hip.cpp
kernel.ptx: kernel.cu lattice.hpp su3.h Makefile
	$(NVCC) -DMILC_COMPLEX -DPRECISION=$(PRECISION) -O3 -ptx -o $@ kernel.cu
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
