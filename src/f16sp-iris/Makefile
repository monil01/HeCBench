IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
HIPFLAGS ?= -O3
EXTRA_CFLAGS ?=
RUN_ARGS ?= 1000
all: main kernel.hip
main: main.cpp config.h
	$(CXX) $(CXXFLAGS) $(EXTRA_CFLAGS) -I$(IRIS_ROOT)/include $< -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
kernel.hip: kernel.hip.cpp config.h
	$(HIPCC) $(HIPFLAGS) $(EXTRA_CFLAGS) --genco $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx
