IRIS_ROOT ?= /tmp/iris
CXX ?= g++
HIPCC ?= /opt/rocm/bin/hipcc
EXTRA_CFLAGS ?=
RUN_ARGS ?= 10
CXXFLAGS ?= -O3 -std=c++17 -fopenmp
all: main kernel.hip
main: main.cpp config.hpp reference.h gpu_lock.hpp
	$(CXX) $(CXXFLAGS) $(EXTRA_CFLAGS) -I$(IRIS_ROOT)/include main.cpp -o $@ -L$(IRIS_ROOT)/build/src/runtime -Wl,-rpath,$(IRIS_ROOT)/build/src/runtime -liris
kernel.hip: kernel.hip.cpp config.hpp
	$(HIPCC) -O3 $(EXTRA_CFLAGS) --genco $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx
