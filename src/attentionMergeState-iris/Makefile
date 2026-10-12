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
	IRIS_ARCHS=HIP ./main 1024 32 128 1000
	IRIS_ARCHS=HIP ./main 1024 64 128 1000
	IRIS_ARCHS=HIP ./main 1024 128 128 1000
	IRIS_ARCHS=HIP ./main 1024 256 128 1000

clean:
	rm -f main kernel.hip
