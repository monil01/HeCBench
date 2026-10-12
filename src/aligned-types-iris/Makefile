IRIS_ROOT ?= /tmp/iris
IRIS_LIB_DIR ?= $(IRIS_ROOT)/build/src/runtime
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall -I$(IRIS_ROOT)/include
LDFLAGS ?= -L$(IRIS_LIB_DIR) -liris
RUN_ARGS ?=

all: main kernel.hip

main: main.cpp types.h Makefile
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(LDFLAGS)

kernel.hip: kernel.hip.cpp types.h Makefile
	$(HIPCC) --genco -O3 -std=c++17 -o $@ kernel.hip.cpp

run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)

clean:
	rm -f main kernel.hip kernel.ptx *.o
