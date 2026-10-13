IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
RUN_ARGS ?= 100 100000
all: main kernel.hip
main: main.cpp reference.cpp utils.cpp utils.h dmem.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp reference.cpp utils.cpp -o $@ $(LDFLAGS)
kernel.hip: kernel.hip.cpp utils.h
	$(HIPCC) -O3 --genco $< -o $@
kernel.ptx: kernel.cu utils.h
	nvcc -O3 -ptx $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
