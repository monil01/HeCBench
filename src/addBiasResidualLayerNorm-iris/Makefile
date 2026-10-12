IRIS_ROOT ?= /tmp/iris
ROCM_ROOT ?= /opt/rocm-7.0.1
CXX ?= g++
HIPCC ?= hipcc
IRIS_CXXFLAGS ?= -O3 -std=c++17 -fopenmp -I$(IRIS_ROOT)/include -I$(ROCM_ROOT)/include
IRIS_LDFLAGS ?= -L$(IRIS_ROOT)/build/src/runtime -Wl,-rpath,$(IRIS_ROOT)/build/src/runtime -liris -lpthread -ldl
RUN_ARGS ?= 10000

.PHONY: all run clean

all: main kernel.hip

main: main.cpp reference.h
	$(CXX) $(IRIS_CXXFLAGS) -o $@ main.cpp $(IRIS_LDFLAGS)

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -O3 -o $@ $<

run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)

clean:
	rm -f main kernel.hip
