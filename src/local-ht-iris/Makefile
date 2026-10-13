IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
RUN_ARGS ?=
all: main kernel.hip
main: main.cpp helper.cpp helper.hpp layout.h dmem.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp helper.cpp -o $@ $(LDFLAGS)
kernel.hip: kernel.hip.cpp
	$(HIPCC) -O3 --genco $< -o $@
kernel.ptx: kernel.cu kernel.hpp
	nvcc -O3 -ptx $< -o $@
run: all
	@if [ -n "$(RUN_ARGS)" ]; then IRIS_ARCHS=HIP ./main $(RUN_ARGS); else bash run-tests.sh; fi
clean:
	rm -f main kernel.hip kernel.ptx *.o test-out.dat
.PHONY: all run clean
