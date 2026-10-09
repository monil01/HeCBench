IRIS_ROOT ?= /tmp/iris
CXX ?= g++
HIPCC ?= hipcc
OPTIMIZE ?= yes
VERIFY ?= yes
FP64 ?= no
RUN_ARGS ?= 100

IRIS_LIBDIR ?= $(IRIS_ROOT)/build/src/runtime
CXXFLAGS ?= -std=c++17 -Wall -I$(IRIS_ROOT)/include
HIPFLAGS ?=
LDFLAGS ?= -L$(IRIS_LIBDIR) -liris -Wl,-rpath,$(IRIS_LIBDIR)

ifeq ($(OPTIMIZE),yes)
  CXXFLAGS += -O3
  HIPFLAGS += -O3
endif

ifeq ($(VERIFY),yes)
  CXXFLAGS += -DVERIFY
endif

ifeq ($(FP64),yes)
  CXXFLAGS += -DDOUBLE_PRECISION
  HIPFLAGS += -DDOUBLE_PRECISION
endif

all: main kernel.hip

main: main.cpp Makefile
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ $<

run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)

clean:
	rm -f main kernel.hip kernel.ptx *.o
