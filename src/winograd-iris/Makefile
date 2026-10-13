INPUT_SIZE ?= 1024
IRIS_ROOT ?= /tmp/iris
IRIS_LIBDIR ?= $(IRIS_ROOT)/build/src/runtime
ifeq ($(origin CXX),default)
CXX = hipcc
endif
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS += -I$(IRIS_ROOT)/include -DMAP_SIZE=$(INPUT_SIZE)
LDFLAGS += -L$(IRIS_LIBDIR) -Wl,-rpath,$(IRIS_LIBDIR)
LDLIBS += -liris
HIPFLAGS += -DMAP_SIZE=$(INPUT_SIZE)
RUN_ARGS ?=
.PHONY: all run clean
all: main kernel.hip
main: main.cpp dmem.hpp utils.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp utils.cpp -o $@ $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp utils.h
	$(HIPCC) $(HIPFLAGS) -O3 -std=c++17 --genco kernel.hip.cpp -o $@
run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip *.o
