IRIS_ROOT ?= /tmp/iris
IRIS_LIBDIR ?= $(IRIS_ROOT)/build/src/runtime
CXX ?= g++
HIPCC ?= hipcc
NVCC ?= nvcc
CXXFLAGS ?= -O3 -std=c++17 -Wall -I$(IRIS_ROOT)/include
LDFLAGS ?= -L$(IRIS_LIBDIR) -liris -Wl,-rpath,$(IRIS_LIBDIR)
RUN_ARGS ?= 10 4096 100

.PHONY: all clean run

all: main kernel.hip

main: main.cpp Makefile
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(LDFLAGS)

kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) --genco -o $@ kernel.hip.cpp

kernel.ptx: kernel.cu Makefile
	$(NVCC) -ptx -o $@ kernel.cu

run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)

clean:
	rm -f main kernel.hip kernel.ptx *.o
