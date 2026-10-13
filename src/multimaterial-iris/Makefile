IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
HOST = main.cpp full_matrix.cpp compact.cpp
all: multimat multimat_F multimat_FL kernel.hip
multimat: $(HOST) dmem.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(HOST) -o $@ $(LDFLAGS)
multimat_F: $(HOST) dmem.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DFUSED $(HOST) -o $@ $(LDFLAGS)
multimat_FL: $(HOST) dmem.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DFUSED -DLINKED $(HOST) -o $@ $(LDFLAGS)
kernel.hip: kernel.hip.cpp kernel_body.h
	$(HIPCC) -O3 --genco $< -o $@
kernel.ptx: kernel.cu kernel_body.h
	nvcc -O3 -ptx $< -o $@
run: all
	IRIS_ARCHS=HIP ./multimat_FL 2000 2000
	IRIS_ARCHS=HIP ./multimat_F 2000 2000
	IRIS_ARCHS=HIP ./multimat 2000 2000
	IRIS_ARCHS=HIP ./multimat_FL 2000 2000 0.3 0.05 0.05
	IRIS_ARCHS=HIP ./multimat_F 2000 2000 0.3 0.05 0.05
	IRIS_ARCHS=HIP ./multimat 2000 2000 0.3 0.05 0.05
clean:
	rm -f multimat multimat_F multimat_FL kernel.hip kernel.ptx *.o
.PHONY: all run clean
