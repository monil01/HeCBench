IRIS_ROOT ?= /tmp/iris
CXX ?= g++
HIPCC ?= hipcc
NVCC ?= nvcc
OPTIMIZE ?= yes
DEBUG ?= no
RUN_ARGS ?=

CXXFLAGS ?= -std=c++17 -Wall -I$(IRIS_ROOT)/include
LDFLAGS ?= -L$(IRIS_ROOT)/build/src/runtime -L$(IRIS_ROOT)/build/lib64 -L$(IRIS_ROOT)/build/lib -Wl,-rpath,$(IRIS_ROOT)/build/src/runtime -Wl,-rpath,$(IRIS_ROOT)/build/lib64 -Wl,-rpath,$(IRIS_ROOT)/build/lib -liris

ifeq ($(OPTIMIZE),yes)
  CXXFLAGS += -O3
endif

ifeq ($(DEBUG),yes)
  CXXFLAGS += -g
  LDFLAGS += -g
endif

all: main kernel.hip

main: main.cpp ../base64e-cuda/reference.h Makefile
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(LDFLAGS)

kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) --genco -o $@ kernel.hip.cpp

kernel.ptx: kernel.cu Makefile
	$(NVCC) -ptx -o $@ kernel.cu

run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)

clean:
	rm -f main kernel.hip kernel.ptx *.o
