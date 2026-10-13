IRIS_ROOT ?= /tmp/iris
IRIS_LIBDIR ?= $(IRIS_ROOT)/build/src/runtime
ifeq ($(origin CXX),default)
CXX = hipcc
endif
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include
LDFLAGS += -L$(IRIS_LIBDIR) -Wl,-rpath,$(IRIS_LIBDIR)
LDLIBS += -liris
HIPFLAGS ?=
RUN_ARGS ?= --i=100
.PHONY: all run clean
all: main kernel.hip
main: main.cpp dmem.hpp decode.h histogram_gmem_atomics.h histogram_smem_atomics.h mersenne.h test_util.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp  -o $@ $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp decode.h histogram_gmem_atomics.h histogram_smem_atomics.h mersenne.h test_util.h
	$(HIPCC) $(HIPFLAGS) -O3 -std=c++17 --genco kernel.hip.cpp -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip *.o
