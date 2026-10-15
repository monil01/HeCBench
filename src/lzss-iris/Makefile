IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
RUN_ARGS ?= -i ../lzss-cuda/QCLOUDf48.bin.f32.errctrl.1e-3
CXXFLAGS ?= -O3 -std=c++17
all: main kernel.hip
main: main.cpp
	$(CXX) $(CXXFLAGS) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include $< -o $@ -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
kernel.hip: kernel.hip.cpp
	$(HIPCC) -O3 --genco $< -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip *.o
.PHONY: all run clean
