IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
RUN_ARGS ?= 7 15 8000 100

CXXFLAGS ?= -O3 -std=c++17 -Wall -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS ?= -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread

all: main kernel.hip

main: main.cpp ../adv-cuda/reference.h Makefile
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) --genco -o $@ $<

run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)

clean:
	rm -f main kernel.hip kernel.ptx *.o
