IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
OPTIMIZE ?= yes
RUN_ARGS ?= 1 8000 100
RUN_ARGS_3 ?= 3 8000 100

CXXFLAGS := -std=c++17 -Wall -Ddfloat=float -Ddlong=int \
  -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
HIPFLAGS :=
LDFLAGS := -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime \
  -liris -ldl -lpthread -llapack -lblas -lgfortran -lm
ifeq ($(OPTIMIZE),yes)
  CXXFLAGS += -O3
  HIPFLAGS += -O3
endif

all: main kernel.hip

main: main.cpp hip_backend.hpp meshBasis.cpp meshBasis.hpp axhelmReference.cpp
	$(CXX) $(CXXFLAGS) -o $@ main.cpp meshBasis.cpp $(LDFLAGS)

kernel.hip: kernel.hip.cpp
	$(HIPCC) $(HIPFLAGS) --genco -o $@ kernel.hip.cpp

kernel.ptx: kernel.cu
	nvcc -ptx -o $@ kernel.cu

run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
	IRIS_ARCHS=HIP ./main $(RUN_ARGS_3)

clean:
	rm -f main kernel.hip kernel.ptx *.o
