IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall
HIPFLAGS ?= -O3 -munsafe-fp-atomics
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
RUN_ARGS ?= 1000 0.5 502 458
LAUNCHER ?=
all: main kernel.hip
main: main.cpp main.h graphics.c resize.c Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ main.cpp $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp main.h extract_kernel.cu prepare_kernel.cu reduce_kernel.cu srad_kernel.cu srad2_kernel.cu compress_kernel.cu Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ kernel.hip.cpp
kernel.ptx: kernel.cu
	nvcc -std=c++17 -ptx -o $@ $<
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
