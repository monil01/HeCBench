IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
all: main kernel.hip
main: main.cpp nfautil.cpp regex.cpp $(wildcard *.h) Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ main.cpp nfautil.cpp regex.cpp $(LDFLAGS) -liris -ldl -lpthread
kernel.hip: kernel.hip.cpp pnfa.cu putil.cu $(wildcard *.h) Makefile
	$(HIPCC) -O3 --genco -o $@ kernel.hip.cpp
kernel.ptx: kernel.cu pnfa.cu putil.cu $(wildcard *.h)
	nvcc -O3 -ptx -o $@ kernel.cu
run: all
	IRIS_ARCHS=HIP bash ./runtests.sh
clean:
	rm -f main kernel.hip kernel.ptx *.o RESULTS
.PHONY: all run clean
