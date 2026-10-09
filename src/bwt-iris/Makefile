IRIS_ROOT ?= /tmp/iris
CXX ?= g++
HIPCC ?= hipcc
OPTIMIZE ?= yes
RUN_ARGS ?= 10000000

IRIS_LIBDIR ?= $(IRIS_ROOT)/build/src/runtime
CXXFLAGS ?= -std=c++17 -Wall -I$(IRIS_ROOT)/include -I.
HIPFLAGS ?=
LDFLAGS ?= -L$(IRIS_LIBDIR) -liris -Wl,-rpath,$(IRIS_LIBDIR)

ifeq ($(OPTIMIZE),yes)
  CXXFLAGS += -O3
  HIPFLAGS += -O3
endif

all: main kernel.hip

main: main.cpp bwt.cpp bwt.hpp Makefile
	$(CXX) $(CXXFLAGS) -o $@ main.cpp bwt.cpp $(LDFLAGS)

kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ kernel.hip.cpp

run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)

clean:
	rm -f main kernel.hip kernel.ptx *.o
