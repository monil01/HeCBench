IRIS_ROOT ?= /tmp/iris
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall -I$(IRIS_ROOT)/include
IRIS_LIB_DIR ?= $(IRIS_ROOT)/build/src/runtime
LDFLAGS ?= -L$(IRIS_LIB_DIR) -liris
RUN_ARGS ?= 1 67108864 100

all: main kernel.hip

main: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $<

run: all
	LD_LIBRARY_PATH=$(IRIS_LIB_DIR):$(LD_LIBRARY_PATH) IRIS_ARCHS=HIP ./main $(RUN_ARGS)

clean:
	rm -f main kernel.hip kernel.ptx *.o
