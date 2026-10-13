IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
NVCC ?= nvcc
CXXFLAGS ?= -O3 -std=c++17 -Wall
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
HIPFLAGS ?= -O3
RUN_ARGS ?=
LAUNCHER ?=

.PHONY: all clean run
all: main kernel.hip

main: main.cpp Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

kernel.hip: kernel.hip.cpp kernels.hpp Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ $<

kernel.ptx: kernel.cu kernels.hpp
	$(NVCC) -O3 -ptx -o $@ $<

run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)

clean:
	rm -f main kernel.hip kernel.ptx *.o
