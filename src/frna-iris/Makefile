IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
RUN_ARGS ?= ../prna-cuda/HIV1-NL43.seq hiv1-nl43.out
CXXFLAGS ?= -O3 -ffast-math -DINT -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
all: main kernel.hip
main: main.cpp frna.cpp fbase.c fparam.c util.c $(wildcard *.h) Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ main.cpp frna.cpp fbase.c fparam.c util.c $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp $(wildcard *.h) Makefile
	$(HIPCC) -O3 -ffast-math -DINT --genco -o $@ kernel.hip.cpp
kernel.ptx: kernel.cu $(wildcard *.h)
	nvcc -O3 -use_fast_math -DINT -ptx -o $@ kernel.cu
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
test: all
	IRIS_ARCHS=HIP ./main RD0260.seq rd0260.out
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run test clean
