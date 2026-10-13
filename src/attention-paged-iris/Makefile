IRIS_ROOT ?= /tmp/iris
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall -I$(IRIS_ROOT)/include
LDFLAGS ?= -L$(IRIS_ROOT)/build/src/runtime -Wl,-rpath,$(IRIS_ROOT)/build/src/runtime -liris

.PHONY: all run clean
all: main kernel.hip

main: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -O3 -std=c++17 -o $@ $<

run: all
	IRIS_ARCHS=HIP ./main 8 32 128 4096 128 100
	IRIS_ARCHS=HIP ./main 8 32 128 4096 1024 100
	IRIS_ARCHS=HIP ./main 8 32 128 4096 8192 100
	IRIS_ARCHS=HIP ./main 8 32 128 4096 131072 100

clean:
	rm -f main kernel.hip
