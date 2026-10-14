IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS += -O3 -std=c++17 -Wall -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include -Isrc/gate -Isrc/parser -Isrc/sim -Isrc/util -Isrc/wave $(EXTRA_CFLAGS)
HIPFLAGS += -O3 -std=c++17 -Isrc/gate -Isrc/parser -Isrc/sim -Isrc/util -Isrc/wave
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
SOURCES = main.cpp src/sim/Simulator.cpp $(wildcard src/gate/*.cpp src/parser/*.cpp src/util/*.cpp src/wave/*.cpp)
HEADERS = $(wildcard src/gate/*.h src/parser/*.h src/sim/*.h src/util/*.h src/wave/*.h)
RUN_ARGS ?= ../logic-resim-cuda/Simulation/toy/intermediate.file ../logic-resim-cuda/Simulation/toy/NV_NVDLA_partition_o_dc_24x33x55_5x5x55x25_int8_input.vcd 0 2972036001 toy-result.saif
all: main kernel.hip
main: $(SOURCES) $(HEADERS) Makefile
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $@ $(LDFLAGS)
kernel.hip: kernel.hip.cpp $(HEADERS) Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ kernel.hip.cpp
kernel.ptx: kernel.cu $(HEADERS)
	nvcc $(HIPFLAGS) -ptx -o $@ kernel.cu
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
