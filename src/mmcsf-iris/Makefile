IRIS_ROOT ?= /tmp/iris
CXX = g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_ROOT)/build/include
LDFLAGS += -L$(IRIS_ROOT)/build/src/runtime -Wl,-rpath,$(IRIS_ROOT)/build/src/runtime -liris -lpthread
RUN_ARGS ?= -i output.tns -m 0 -R 32 -f 128 -w 4
all: main kernel.hip
main: main.cpp mttkrp_gpu.cpp mttkrp_cpu.cpp util.h mttkrp_gpu.h mttkrp_cpu.h dmem.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) main.cpp mttkrp_gpu.cpp mttkrp_cpu.cpp -o $@ $(LDFLAGS)
kernel.hip: kernel.hip.cpp
	$(HIPCC) -O3 --genco $< -o $@
output.tns: generate_tensor.py
	python3 generate_tensor.py output.tns 2%4096 2%4096 2%4096 2%512
run: all output.tns
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
test: all
	IRIS_ARCHS=HIP ./main -i toy.tns -m 0 -R 32 -f 128 -w 1
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run test clean
