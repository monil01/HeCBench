IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX = hipcc
HIPCC ?= hipcc
IRIS_NSTREAMS ?= 16
OPTIMIZE ?= yes
LAUNCHER ?=
RUN_ARGS ?=

CXXFLAGS := -std=c++17 -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include -Wall
LDFLAGS := -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread

ifeq ($(OPTIMIZE),yes)
  CXXFLAGS += -O3
endif

all: main kernel.hip

main: main.cpp Makefile
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(LDFLAGS)

kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) -O3 --genco $< -o $@

runtime_default_test: runtime_default_test.cpp Makefile
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

run: all
	IRIS_ARCHS=HIP IRIS_ASYNC=1 IRIS_NSTREAMS=$(IRIS_NSTREAMS) $(LAUNCHER) ./main $(RUN_ARGS)

clean:
	rm -f main kernel.hip runtime_default_test *.o
