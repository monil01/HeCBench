IRIS_ROOT ?= /tmp/iris
CXX = hipcc
HIPCC = hipcc
CXXFLAGS ?= -O3 -std=c++17 -I$(IRIS_ROOT)/include
LDFLAGS ?= -L$(IRIS_ROOT)/build/src/runtime -Wl,-rpath,$(IRIS_ROOT)/build/src/runtime -liris

all: main kernel.hip

main: main.cpp cpu_reference.h exception.h layout.cuh math.cuh pos_enc.cuh utils.cuh utils.h vec_dtypes.cuh
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(LDFLAGS)

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $<

run: all
	IRIS_ARCHS=HIP ./main 100

clean:
	rm -f main kernel.hip
