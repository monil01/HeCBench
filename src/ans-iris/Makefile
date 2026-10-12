IRIS_ROOT ?= /tmp/iris
IRIS_LIB_DIR ?= $(IRIS_ROOT)/build/src/runtime
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Iinclude -I$(IRIS_ROOT)/include
LDFLAGS ?= -L$(IRIS_LIB_DIR) -liris
RUN_ARGS ?= 1

SRC_FILES := $(filter-out src/main.cu src/multians_gpu_decoder.cu,$(wildcard src/*.cc))
OBJ_FILES := $(patsubst src/%.cc,build/%.o,$(SRC_FILES))

all: main kernel.hip

main: main.cpp $(OBJ_FILES) Makefile
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(OBJ_FILES) $(LDFLAGS)

build/%.o: src/%.cc $(wildcard include/*.h) Makefile
	mkdir -p build
	$(CXX) $(CXXFLAGS) -c -o $@ $<

kernel.hip: kernel.hip.cpp $(wildcard include/*.h) Makefile
	$(HIPCC) --genco -O3 --std=c++17 -Iinclude -o $@ kernel.hip.cpp

run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)

clean:
	rm -rf main kernel.hip kernel.ptx build *.o
