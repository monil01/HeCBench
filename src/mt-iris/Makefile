IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall
HIPFLAGS ?= -O3
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
RUN_ARGS ?= 1000
LAUNCHER ?=
all: main kernel.hip
main: main.cpp MT_gold.cpp genmtrand.cpp MT.h dci.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp MT_gold.cpp genmtrand.cpp -o $@ $(LDFLAGS)
kernel.hip: kernel.hip.cpp MT.h Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ kernel.hip.cpp
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
