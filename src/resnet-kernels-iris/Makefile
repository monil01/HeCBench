IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
HOST = main.cpp util.cpp Kernel128_one.cpp Kernel256_one.cpp Kernel128_winograd.cpp Kernel256_winograd.cpp
all: main kernel.hip
main: $(HOST) $(wildcard *.h) Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $(HOST) $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) -O3 --genco -o $@ kernel.hip.cpp
kernel.ptx: kernel.cu
	nvcc -O3 -ptx -o $@ kernel.cu
data/input_14_1_128.bin:
	mkdir -p data
	python3 data_generator.py
run: all data/input_14_1_128.bin
	for mode in 0 1 2 3 4 5; do IRIS_ARCHS=HIP ./main $$mode 100 || exit $$?; done
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean

