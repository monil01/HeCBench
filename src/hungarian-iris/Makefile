IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
EXTRA_CFLAGS ?=
RUN_ARGS ?= output.txt
CXXFLAGS += -O3 -std=c++17 -Wall -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include $(EXTRA_CFLAGS)
HIPFLAGS += -O3 -std=c++17 $(EXTRA_CFLAGS)
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
all: main kernel.hip
main: main.cpp state.h Makefile
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(LDFLAGS)
kernel.hip: kernel.hip.cpp state.h Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ kernel.hip.cpp
kernel.ptx: kernel.cu state.h
	nvcc $(EXTRA_CFLAGS) -O3 -ptx -o $@ kernel.cu
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
