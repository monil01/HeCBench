IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
NVCC ?= nvcc
OPTIMIZE ?= yes
LAUNCHER ?=
RUN_ARGS ?= 1048576 10

CXXFLAGS := -std=c++17 -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include -Wall
HIPFLAGS :=
LDFLAGS := -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread

ifeq ($(OPTIMIZE),yes)
  CXXFLAGS += -O3 -ffast-math
  HIPFLAGS += -O3 -ffast-math
endif

all: main kernel.hip

main: main.cpp Makefile
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(LDFLAGS)

kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ kernel.hip.cpp

kernel.ptx: kernel.cu Makefile
	$(NVCC) -ptx -o $@ kernel.cu

run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)

clean:
	rm -f main kernel.hip kernel.ptx *.o
