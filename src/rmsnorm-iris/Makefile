IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS ?= -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
RUN_ARGS ?= 1 12288 1000

all: main kernel.hip

main: main.cpp Makefile
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(LDFLAGS)

kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) --genco -O3 -std=c++17 -o $@ kernel.hip.cpp

kernel.ptx: kernel.cu Makefile
	nvcc -ptx -O3 -std=c++17 -o $@ kernel.cu

run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)

clean:
	rm -f main kernel.hip kernel.ptx *.o
