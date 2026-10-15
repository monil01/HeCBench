IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
OPTIMIZE ?= yes
LAUNCHER ?=
RUN_ARGS ?= 1000

CXXFLAGS := -std=c++17 -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include -I/opt/rocm/include -Wall
HIPFLAGS :=
LDFLAGS := -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread

ifeq ($(OPTIMIZE),yes)
  CXXFLAGS += -O3
  HIPFLAGS += -O3
endif

all: main kernel.hip

main: main.cpp Makefile
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(LDFLAGS)

kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ kernel.hip.cpp

kernel.ptx: kernel.cu Makefile
	nvcc -ptx -o $@ kernel.cu

run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)

runtime_test: runtime_test.cpp Makefile
	$(CXX) $(CXXFLAGS) -D__HIP_PLATFORM_AMD__ -o $@ runtime_test.cpp $(LDFLAGS) -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -lamdhip64

diagnostics: finegrain_probe system_ops_probe

finegrain_probe: finegrain_probe.hip.cpp
	$(HIPCC) -O3 -std=c++17 -o $@ $<

system_ops_probe: system_ops_probe.hip.cpp
	$(HIPCC) -O3 -std=c++17 -o $@ $< -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -lhsa-runtime64

clean:
	rm -f main kernel.hip kernel.ptx runtime_test finegrain_probe system_ops_probe *.o
