IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
EXTRA_CFLAGS ?=
RUN_ARGS ?= 1000
#
all: main kernel.hip
main: main.cpp constants.h rand_helpers.h reference.h Makefile
	$(CXX) $(CXXFLAGS) $(EXTRA_CFLAGS) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include $< -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
kernel.hip: kernel.hip.cpp constants.h Makefile
	$(HIPCC) -O3 $(EXTRA_CFLAGS) --genco $< -o $@
kernel.ptx: kernel.cu
	nvcc -O3 -ptx $< -o $@
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
