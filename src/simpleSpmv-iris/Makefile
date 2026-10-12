IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
HIPFLAGS ?= -O3 -std=c++17
LDFLAGS ?= -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
RUN_ARGS ?= 4096 512 1

all: main kernel.hip

main: main.cpp utils.cpp mv.h
	$(CXX) $(CXXFLAGS) -o $@ main.cpp utils.cpp $(LDFLAGS)

kernel.hip: kernel.hip.cpp
	$(HIPCC) $(HIPFLAGS) --genco -o $@ $<

run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)

clean:
	rm -f main kernel.hip kernel.ptx *.o
