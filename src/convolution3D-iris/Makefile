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
RUN_ARGS ?=
.PHONY: all run clean
all: main kernel.hip
main: main.cpp dmem.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp  -o $@ $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp
	$(HIPCC) $(HIPFLAGS) -O3 -std=c++17 --genco kernel.hip.cpp -o $@
run: all
ifneq ($(strip $(RUN_ARGS)),)
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)
else
	IRIS_ARCHS=HIP ./main 32 1 6 32 32 5 100
	IRIS_ARCHS=HIP ./main 32 6 16 14 14 5 100
	IRIS_ARCHS=HIP ./main 32 96 256 26 26 5 100
endif
clean:
	rm -f main kernel.hip *.o
