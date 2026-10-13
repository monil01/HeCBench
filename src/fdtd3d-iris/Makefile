IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall
HIPFLAGS ?= -O3 -munsafe-fp-atomics
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
RUN_ARGS ?= --dimx=192 --dimy=184 --timesteps=90
LAUNCHER ?=
all: main kernel.hip
main: main.cpp FDTD3dGPU.cpp FDTD3dReference.cpp shrUtils.cpp cmd_arg_reader.cpp FDTD3dGPU.h FDTD3dReference.h shrUtils.h cmd_arg_reader.h exception.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ main.cpp FDTD3dGPU.cpp FDTD3dReference.cpp shrUtils.cpp cmd_arg_reader.cpp $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp FDTD3dGPU.h Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ kernel.hip.cpp
kernel.ptx: kernel.cu
	nvcc -std=c++17 -ptx -o $@ $<
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
