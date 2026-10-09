IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
OPTIMIZE ?= yes
LAUNCHER ?=
RUN_ARGS ?= 536870912

CXXFLAGS := -std=c++17 -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include -Wall
LDFLAGS := -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread

ifeq ($(OPTIMIZE),yes)
  CXXFLAGS += -O3
endif

all: main

main: main.cpp Makefile
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(LDFLAGS)

run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)

clean:
	rm -f main *.o
