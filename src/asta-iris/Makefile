IRIS_ROOT ?= /tmp/iris
CXX ?= g++
HIPCC ?= hipcc
IRIS_INC ?= $(IRIS_ROOT)/include
IRIS_LIB ?= $(IRIS_ROOT)/build/src/runtime
CXXFLAGS ?= -O3 -std=c++17 -Wall
CXXFLAGS += -I$(IRIS_INC) $(EXTRA_CXXFLAGS)
LDFLAGS ?= -L$(IRIS_LIB) -Wl,-rpath,$(IRIS_LIB) -liris
RUN_ARGS ?=

.PHONY: all run clean

all: main kernel.hip

main: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco $(HIPCC_FLAGS) -o $@ $<

run: all
	IRIS_ARCHS=HIP LD_LIBRARY_PATH=$(IRIS_LIB):$${LD_LIBRARY_PATH} ./main $(RUN_ARGS)

clean:
	rm -f main kernel.hip kernel.ptx *.o
