IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
RUN_ARGS ?= HIV1-NL43.seq
CXXFLAGS ?= -O3 -ffast-math -DFLOAT -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
all: main kernel.hip
main: main.cpp prna.cpp base.c param.c util.c $(wildcard *.h) Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ main.cpp prna.cpp base.c param.c util.c $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp $(wildcard *.h) Makefile
	$(HIPCC) -O3 -ffast-math -DFLOAT --genco -o $@ kernel.hip.cpp
kernel.ptx: kernel.cu $(wildcard *.h)
	nvcc -O3 -use_fast_math -DFLOAT -ptx -o $@ kernel.cu
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean

