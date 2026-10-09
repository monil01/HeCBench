IRIS_ROOT ?= /tmp/iris
CXX ?= g++
HIPCC ?= hipcc
OPTIMIZE ?= yes
RUN_ARGS ?= 100000 2097152 0
TEST_ARGS ?= 1000 8192 1

IRIS_LIBDIR ?= $(IRIS_ROOT)/build/src/runtime
CXXFLAGS ?= -std=c++17 -fopenmp -Wall -I$(IRIS_ROOT)/include
HIPFLAGS ?=
LDFLAGS ?= -L$(IRIS_LIBDIR) -liris -Wl,-rpath,$(IRIS_LIBDIR) -fopenmp

ifeq ($(OPTIMIZE),yes)
  CXXFLAGS += -O3
  HIPFLAGS += -O3
endif

all: main kernel.hip

main: main.cpp Makefile
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(LDFLAGS)

kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ kernel.hip.cpp

test: all
	IRIS_ARCHS=HIP ./main $(TEST_ARGS)

run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)

clean:
	rm -f main kernel.hip kernel.ptx *.o
