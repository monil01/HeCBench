IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread
HIPFLAGS ?= -O3
LAUNCHER ?=
RUN_ARGS ?= 
OUTPUT ?= no
ifeq ($(OUTPUT),yes)
CPPFLAGS += -DBENCH_PRINT
endif
.PHONY: all clean run
all: main kernel.hip
main: main.cpp iris_support.hpp helper_string.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)
kernel.hip: kernel.hip.cpp kernels.hpp Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ $<
kernel.ptx: kernel.cu kernels.hpp
	nvcc -O3 -ptx -o $@ $<
run: all
	if test -n "$(RUN_ARGS)"; then IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS); else \
	  IRIS_ARCHS=HIP $(LAUNCHER) ./main -float -keysonly -n=536870912 && \
	  IRIS_ARCHS=HIP $(LAUNCHER) ./main -float -n=536870912 && \
	  IRIS_ARCHS=HIP $(LAUNCHER) ./main -keysonly -n=536870912 && \
	  IRIS_ARCHS=HIP $(LAUNCHER) ./main -n=536870912; fi
clean:
	rm -f main kernel.hip kernel.ptx *.o
