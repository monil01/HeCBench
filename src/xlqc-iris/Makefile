IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
GSL_INC ?=
GSL_LIB ?= -lgsl -lgslcblas
RUN_ARGS ?= sp
DATA_DIR ?= .
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I../xlqc-cuda -I../xlqc-cuda/int_lib -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include $(GSL_INC)
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread $(GSL_LIB)
all: main kernel.hip
main: main.cpp Makefile
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) main.cpp ../xlqc-cuda/basis.cc ../xlqc-cuda/scf.cc ../xlqc-cuda/int_lib/crys.cc ../xlqc-cuda/int_lib/cints.cc $(LDFLAGS) -o $@
kernel.hip: kernel.hip.cpp
	$(HIPCC) -O3 -I. -I../xlqc-cuda -I../xlqc-hip --genco kernel.hip.cpp -o $@
run: all
	cd $(DATA_DIR) && IRIS_ARCHS=HIP IRIS_KERNEL_BIN_HIP=$(abspath kernel.hip) $(abspath main) $(RUN_ARGS)
	cd $(DATA_DIR) && IRIS_ARCHS=HIP IRIS_KERNEL_BIN_HIP=$(abspath kernel.hip) $(abspath main) dp
clean:
	rm -f main kernel.hip kernel.ptx *.o
