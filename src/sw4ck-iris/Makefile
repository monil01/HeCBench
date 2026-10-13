IRIS_ROOT ?= /tmp/iris
IRIS_LIBDIR ?= $(IRIS_ROOT)/build/src/runtime
CXX = g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
HIPFLAGS ?= -O3 -std=c++17
RUN_ARGS ?= ../sw4ck-cuda/sw4ck.in 100
.PHONY: all run clean
all: main kernel.hip
main: main.cpp utils.cpp utils.h curvilinear4sg.cpp curvilinear4sg.h
	$(CXX) $(CXXFLAGS) -I$(IRIS_ROOT)/include main.cpp curvilinear4sg.cpp -o $@ -L$(IRIS_LIBDIR) -Wl,-rpath,$(IRIS_LIBDIR) -liris
kernel.hip: kernel.hip.cpp kernel1.cpp kernel2.cpp kernel3.cpp kernel4.cpp kernel5.cpp utils.h curvilinear4sg.h
	$(HIPCC) $(HIPFLAGS) --genco kernel.hip.cpp -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
