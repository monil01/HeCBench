IRIS_ROOT ?= /tmp/iris
IRIS_LIBDIR ?= $(IRIS_ROOT)/build/src/runtime
ifeq ($(origin CXX),default)
CXX = hipcc
endif
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -fopenmp
CPPFLAGS += -I$(IRIS_ROOT)/include
LDFLAGS += -L$(IRIS_LIBDIR) -Wl,-rpath,$(IRIS_LIBDIR)
LDLIBS += -liris
HIPFLAGS ?=
RUN_ARGS ?=
.PHONY: all run clean
all: main kernel.hip
main: main.cpp dmem.hpp muon_common.h reference.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp  -o $@ $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp muon_common.h reference.h
	$(HIPCC) $(HIPFLAGS) -O3 -std=c++17 --genco kernel.hip.cpp -o $@
run: all
ifneq ($(strip $(RUN_ARGS)),)
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
else
	IRIS_ARCHS=HIP ./main 3072 768 10 1
	IRIS_ARCHS=HIP ./main 768 3072 10 1
	IRIS_ARCHS=HIP ./main 3072 768 10000 0
	IRIS_ARCHS=HIP ./main 768 3072 10000 0
endif
clean:
	rm -f main kernel.hip *.o
