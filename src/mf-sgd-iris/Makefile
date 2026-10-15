IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
HOSTCXX ?= hipcc
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall
HIPFLAGS ?= -O3
TRAIN ?= ../mf-sgd-hip/data/netflix/netflix_train.bin
MODEL ?= $(notdir $(TRAIN)).model
RUN_ARGS ?= -g 1 -l 0.05 -a 0.08 -b 0.3 -u 1 -v 1 -x 1 -y 1 -s 750 -k 128 -t 20 $(TRAIN) $(MODEL)
INCLUDES = -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LIBS = -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
all: main kernel.hip
main: main.cpp sgd.cpp sgd.h node.h dmem_adapter.cpp dmem_adapter.hpp
	$(HOSTCXX) $(CXXFLAGS) $(INCLUDES) main.cpp sgd.cpp dmem_adapter.cpp $(LIBS) -o $@
kernel.hip: kernel.hip.cpp node.h
	$(HIPCC) $(HIPFLAGS) --genco kernel.hip.cpp -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip *.o
.PHONY: all run clean
