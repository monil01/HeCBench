IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
OPTIMIZE ?= yes
LAUNCHER ?=
RUN_ARGS ?= 1000000 1000 100

CXXFLAGS := -std=c++17 -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include -Wall
HIPFLAGS :=
LDFLAGS := -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread

ifeq ($(OPTIMIZE),yes)
  CXXFLAGS += -O3
  HIPFLAGS += -O3
endif

all: main kernel.hip

main: main.cpp sobol_gold.cpp sobol_primitives.cpp sobol.h sobol_gold.h sobol_primitives.h Makefile
	$(CXX) $(CXXFLAGS) -o $@ main.cpp sobol_gold.cpp sobol_primitives.cpp $(LDFLAGS)

kernel.hip: kernel.hip.cpp sobol.h Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ kernel.hip.cpp

kernel.ptx: kernel.cu sobol.h Makefile
	nvcc -ptx -o $@ kernel.cu

run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)

clean:
	rm -f main kernel.hip kernel.ptx *.o
