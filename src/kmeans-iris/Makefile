IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
RUN_ARGS ?= -r -n 5 -m 15 -l 10 -o -i ../data/kmeans/kdd_cup
HOST = main.cpp cluster.cpp getopt.cpp read_input.cpp rmse.cpp
all: main kernel.hip
main: $(HOST) kmeans.h getopt.h dmem.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(HOST) -o $@ $(LDFLAGS)
kernel.hip: kernel.hip.cpp
	$(HIPCC) -O3 --genco $< -o $@
kernel.ptx: kernel.cu
	nvcc -O3 -ptx $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
