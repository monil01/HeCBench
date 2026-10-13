IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
NVCC ?= nvcc
RUN_ARGS ?= 65536
LAUNCHER ?=
DUMP ?= no
CXXFLAGS ?= -O3 -std=c++17 -Wall
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
ifeq ($(DUMP),yes)
CPPFLAGS += -DOUTPUT
endif

.PHONY: all run clean
all: main kernel.hip
main: main.cpp backprop.cpp facetrain.cpp imagenet.cpp backprop.h reference.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp backprop.cpp facetrain.cpp imagenet.cpp -o $@ $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp backprop.h Makefile
	$(HIPCC) -O3 --genco -o $@ kernel.hip.cpp
kernel.ptx: kernel.cu backprop.h Makefile
	$(NVCC) -O3 -ptx -o $@ kernel.cu
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
