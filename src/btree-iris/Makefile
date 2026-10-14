IRIS_ROOT ?= /tmp/iris
CXX ?= g++
HIPCC ?= /opt/rocm/bin/hipcc
RUN_ARGS ?= 50000000 50000000
CXXFLAGS ?= -O3 -std=c++17
all: main kernel.hip
main: main.cpp
	$(CXX) $(CXXFLAGS) -I$(IRIS_ROOT)/include $< -o $@ -L$(IRIS_ROOT)/build/src/runtime -Wl,-rpath,$(IRIS_ROOT)/build/src/runtime -liris
kernel.hip: kernel.hip.cpp map_warps.hpp
	$(HIPCC) -O3 --genco $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip
