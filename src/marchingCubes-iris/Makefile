IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
RUN_ARGS ?= 100
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
all: main kernel.hip
main: main.cpp iris_host.h params.h tables.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ main.cpp $(LDFLAGS) -liris -ldl -lpthread
kernel.hip: kernel.hip.cpp tables.h Makefile
	$(HIPCC) -O3 --genco -o $@ kernel.hip.cpp
kernel.ptx: kernel.cu tables.h
	nvcc -O3 -ptx -o $@ kernel.cu
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
