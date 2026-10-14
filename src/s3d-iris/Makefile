IRIS_ROOT ?= /tmp/iris
CXX ?= g++
HIPCC ?= /opt/rocm/bin/hipcc
CXXFLAGS ?= -O3 -std=c++17
RUN_ARGS ?= -q -n 100 -s 1
all: main kernel.hip
main: main.cpp Option.cpp OptionParser.cpp Option.h OptionParser.h Utility.h config.h gpu_lock.hpp
	$(CXX) $(CXXFLAGS) -I$(IRIS_ROOT)/include main.cpp Option.cpp OptionParser.cpp -o $@ -L$(IRIS_ROOT)/build/src/runtime -Wl,-rpath,$(IRIS_ROOT)/build/src/runtime -liris
kernel.hip: kernel.hip.cpp $(wildcard device/*.h)
	$(HIPCC) -O3 --genco $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip
