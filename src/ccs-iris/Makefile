IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
RUN_ARGS ?= -t 0.9 -i ../ccs-cuda/Data_Constant_100_1_bicluster.txt -o Output.txt -m 50 -p 1 -g 100.0 -r 100
all: main kernel.hip
main: main.cpp ccs.h matrixsize.c readgene.c pair_cor.c bicluster_pair_score.c merge_bicluster.c print_bicluster.c
	$(CXX) $(CXXFLAGS) -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include -o $@ main.cpp -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
kernel.hip: kernel.hip.cpp ccs.h
	$(HIPCC) -O3 --genco -o $@ kernel.hip.cpp
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip kernel.ptx *.o
