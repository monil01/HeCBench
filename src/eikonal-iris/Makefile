IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
RUN_ARGS ?= -s 512
HOST = main.cpp StructuredEikonal.cpp fim.cpp timer.cpp
all: main kernel.hip
main: $(HOST) $(wildcard *.h) Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(HOST) -o $@ $(LDFLAGS)
kernel.hip: kernel.hip.cpp kernel.h common_def.h
	$(HIPCC) -O3 --genco $< -o $@
kernel.ptx: kernel.cu kernel.h common_def.h
	nvcc -O3 -ptx $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o output.nrrd
.PHONY: all run clean
