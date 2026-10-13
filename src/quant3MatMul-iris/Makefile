IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
NVCC ?= nvcc
RUN_ARGS ?= 1000
LAUNCHER ?=
CXXFLAGS ?= -O3 -std=c++17 -Wall
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
.PHONY: all run clean
all: main kernel.hip
main: main.cpp iris_support.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp -o $@ $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) -O3 --genco -o $@ kernel.hip.cpp
kernel.ptx: kernel.cu Makefile
	$(NVCC) -O3 -ptx -o $@ kernel.cu
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
