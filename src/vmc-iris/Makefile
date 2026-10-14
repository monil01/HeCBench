IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
RUN_ARGS ?= 100
CXXFLAGS ?= -O3 -std=c++17
all: main kernel.hip
main: main.cpp Makefile
	$(CXX) $(CXXFLAGS)  -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread -o $@
kernel.hip: kernel.hip.cpp
	$(HIPCC) -O3 --genco kernel.hip.cpp -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
