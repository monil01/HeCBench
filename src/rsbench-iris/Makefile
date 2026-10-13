IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = g++
HIPCC ?= hipcc
RUN_ARGS ?= -s large -m event
LAUNCHER ?=
all: main kernel.hip
main: main.cpp simulation.cpp common.cpp init.cpp io.cpp material.cpp utils.cpp rsbench.h backend.h
	$(CXX) -O3 -std=c++17 -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include main.cpp simulation.cpp common.cpp init.cpp io.cpp material.cpp utils.cpp -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -lpthread -ldl
kernel.hip: kernel.hip.cpp rsbench.h kernel_body.h
	$(HIPCC) -O3 --genco kernel.hip.cpp -o $@
kernel.ptx: kernel.cu rsbench.h kernel_body.h
	nvcc -O3 -ptx kernel.cu -o $@
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
