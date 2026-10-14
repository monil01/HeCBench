IRIS_ROOT ?= /tmp/iris
IRIS_LIB_DIR ?= $(IRIS_ROOT)/build/src/runtime
CXX = hipcc
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Iinclude -I$(IRIS_ROOT)/include
LDFLAGS ?= -L$(IRIS_LIB_DIR) -Wl,-rpath,$(IRIS_LIB_DIR) -liris
RUN_ARGS ?= 10

SRC_FILES := $(filter-out src/main.cu src/multians_gpu_decoder.cu,$(wildcard src/*.cc))
OBJ_FILES := $(patsubst src/%.cc,build/%.o,$(SRC_FILES))

all: main kernel.hip

main: build/main.o $(OBJ_FILES) Makefile
	$(CXX) -o $@ build/main.o $(OBJ_FILES) $(LDFLAGS)

build/main.o: main.cpp hip_backend.hpp $(wildcard include/*.h) Makefile
	mkdir -p build
	$(CXX) $(CXXFLAGS) -c -o $@ main.cpp

build/%.o: src/%.cc $(wildcard include/*.h) Makefile
	mkdir -p build
	$(CXX) $(CXXFLAGS) -c -o $@ $<

kernel.hip: kernel.hip.cpp $(wildcard include/*.h) Makefile
	$(HIPCC) --genco -O3 --std=c++17 -Iinclude -o $@ kernel.hip.cpp

run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)

clean:
	rm -rf main kernel.hip kernel.ptx build *.o
