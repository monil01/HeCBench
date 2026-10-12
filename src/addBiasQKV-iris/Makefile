IRIS_ROOT ?= /tmp/iris
CXX ?= g++
HIPCC ?= hipcc
IRIS_ARCH ?=
FP8_FNUZ ?= 1
CXXFLAGS ?= -O3 -std=c++17 -Wall -DIRIS_FP8_FNUZ=$(FP8_FNUZ) -I$(IRIS_ROOT)/include
LDFLAGS ?= -L$(IRIS_ROOT)/build/src/runtime -Wl,-rpath,$(IRIS_ROOT)/build/src/runtime -liris

.PHONY: all run clean

all: main kernel.hip

main: main.cpp ../addBiasQKV-cuda/reference.h
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(LDFLAGS)

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco $(if $(IRIS_ARCH),--offload-arch=$(IRIS_ARCH),) -o $@ $<

run: all
	IRIS_ARCHS=HIP ARCHS=hip ./main

clean:
	rm -f main kernel.hip
