IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -Iinclude -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
RUN_ARGS ?= ../minimap2-sycl/in-1k.txt kernel-out.txt
all: main kernel.hip
main: main.cpp $(wildcard src/*.cpp include/*.h) Makefile
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) main.cpp src/*.cpp -o $@ $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp $(wildcard include/*.h) Makefile
	$(HIPCC) -O3 -Iinclude --genco $< -o $@
kernel.ptx: kernel.cu $(wildcard include/*.h)
	nvcc -O3 -Iinclude -ptx $< -o $@
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
