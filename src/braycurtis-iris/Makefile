IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
PYTHON ?= python3
RUN_ARGS ?=
CXXFLAGS ?= -O3 -std=c++17
all: main kernel.hip
main: main.cpp
	$(CXX) $(CXXFLAGS) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include $< -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread -o $@
kernel.hip: kernel.hip.cpp
	$(HIPCC) -O3 --genco $< -o $@
run: all
	IRIS_ARCHS=HIP $(PYTHON) run.py $(RUN_ARGS)
clean:
	rm -f main kernel.hip *.o
.PHONY: all run clean
