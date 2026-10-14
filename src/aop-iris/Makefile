IRIS_ROOT ?= /tmp/iris
CXX ?= g++
HIPCC ?= /opt/rocm/bin/hipcc
RUN_ARGS ?= -runs 100
CXXFLAGS ?= -O3 -std=c++17
all: main kernel.hip
main: main.cpp payoff.hpp gpu_lock.hpp
	$(CXX) $(CXXFLAGS) -I$(IRIS_ROOT)/include $< -o $@ -L$(IRIS_ROOT)/build/src/runtime -Wl,-rpath,$(IRIS_ROOT)/build/src/runtime -liris
kernel.hip: kernel.hip.cpp device.hpp
	$(HIPCC) -O3 --genco $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip
