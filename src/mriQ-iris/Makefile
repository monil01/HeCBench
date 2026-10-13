IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = hipcc
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -ffast-math -std=c++17 -Wall
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
HIPFLAGS ?= -O3 -ffast-math
LAUNCHER ?=
RUN_ARGS ?= ../mriQ-cuda/datasets/128x128x128/input/128x128x128.bin output_128
OUTPUT ?= no
ifeq ($(OUTPUT),yes)
CPPFLAGS += -DBENCH_PRINT
endif
.PHONY: all clean run
all: main kernel.hip
main: main.cpp iris_support.hpp config.hpp file.cc file.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ main.cpp file.cc $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp kernels.hpp config.hpp Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ $<
kernel.ptx: kernel.cu kernels.hpp
	nvcc -O3 -ptx -o $@ $<
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
