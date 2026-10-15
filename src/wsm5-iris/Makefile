IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
RUN_ARGS ?= 10
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
all: main kernel.hip
main: main.cpp utils.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ main.cpp $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp kernel.h constants.h spt.h Makefile
	$(HIPCC) -DMKX=4 -O3 --genco -o $@ kernel.hip.cpp
kernel.ptx: kernel.cu kernel.h constants.h spt.h
	nvcc -DMKX=4 -O3 -ptx -o $@ kernel.cu
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
