IRIS_ROOT ?= /tmp/iris
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -I$(IRIS_ROOT)/include
LDFLAGS ?= -L$(IRIS_ROOT)/build/src/runtime -Wl,-rpath,$(IRIS_ROOT)/build/src/runtime -liris

.PHONY: all run clean

all: main kernel.hip

main: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $<

run: all
	IRIS_ARCHS=HIP IRIS_KERNEL_BIN_HIP=./kernel.hip ./main

clean:
	rm -f main kernel.hip kernel.ptx *.o
