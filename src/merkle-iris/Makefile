IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
VERIFY ?= no
CXXFLAGS ?= -O3 -std=c++17
ifeq ($(VERIFY),yes)
CPPFLAGS += -DDEBUG
endif
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
all: main kernel.hip
main: main.cpp bench_merkle_tree.cpp parameters.cpp $(wildcard *.hpp) types.h iris_host.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ main.cpp bench_merkle_tree.cpp parameters.cpp $(LDFLAGS) -liris -ldl -lpthread
kernel.hip: kernel.hip.cpp kernel_phases.h ff_p.cu rescue_prime.cu $(wildcard *.hpp) types.h Makefile
	$(HIPCC) -O3 --genco -o $@ kernel.hip.cpp
kernel.ptx: kernel.cu kernel_phases.h ff_p.cu rescue_prime.cu $(wildcard *.hpp) types.h
	nvcc -O3 -ptx -o $@ kernel.cu
run: all
	IRIS_ARCHS=HIP ./main
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
