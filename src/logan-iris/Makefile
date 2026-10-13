IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
RUN_ARGS ?= inputs/100k.txt 17 100 1
CXXFLAGS ?= -O3 -std=c++17 -fopenmp -DADAPTABLE
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
all: main kernel.hip
main: main.cpp logan_functions.cpp seed.cpp score.cpp $(wildcard *.cuh) host_qualifiers.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ main.cpp logan_functions.cpp seed.cpp score.cpp $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp $(wildcard *.cuh) Makefile
	$(HIPCC) -O3 -DADAPTABLE --genco -o $@ kernel.hip.cpp
kernel.ptx: kernel.cu $(wildcard *.cuh)
	nvcc -O3 -DADAPTABLE -ptx -o $@ kernel.cu
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean

