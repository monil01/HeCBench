IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = g++
HIPCC ?= hipcc
RUN_ARGS ?= Face.pgm info.txt class.txt Output-gpu.pgm
CPU_RUN_ARGS ?= Face.pgm info.txt class.txt Output-cpu.pgm
LAUNCHER ?=
HOST_SOURCES = main.cpp haar.cpp image.cpp stdio-wrapper.cpp rectangles.cpp
FLAGS = -O3 -std=c++17 -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LIBS = -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -lpthread -ldl
all: main main-cpu kernel.hip
main: $(HOST_SOURCES) $(wildcard *.h)
	$(CXX) $(FLAGS) -DGPU $(HOST_SOURCES) -o $@ $(LIBS)
main-cpu: $(HOST_SOURCES) $(wildcard *.h)
	$(CXX) $(FLAGS) $(HOST_SOURCES) -o $@ $(LIBS)
kernel.hip: kernel.hip.cpp
	$(HIPCC) -O3 --genco kernel.hip.cpp -o $@
kernel.ptx: kernel.cu
	nvcc -O3 -ptx kernel.cu -o $@
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main-cpu $(CPU_RUN_ARGS)
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main main-cpu kernel.hip kernel.ptx *.o
