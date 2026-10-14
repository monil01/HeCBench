IRIS_ROOT ?= /tmp/iris
HIP_ROOT ?= /opt/rocm
CXX = g++
HIPCC ?= $(HIP_ROOT)/bin/hipcc
CXXFLAGS ?= -O3 -std=c++17
EXTRA_CFLAGS ?=
RUN_ARGS ?= 1000
.PHONY: all run clean
all: main kernel.hip
main: main.cpp reference.h Makefile
	$(CXX) $(CXXFLAGS) $(EXTRA_CFLAGS) -I$(IRIS_ROOT)/include $< -o $@ -L$(IRIS_ROOT)/build/src/runtime -Wl,-rpath,$(IRIS_ROOT)/build/src/runtime -liris -pthread
kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) -O3 -std=c++17 --genco $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
