IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall
HIPFLAGS ?= -O3
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime -liris -ldl -lpthread
RUN_ARGS ?=
LAUNCHER ?=
all: main kernel.hip
main: main.cpp utils.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp -o $@ $(LDFLAGS)
kernel.hip: kernel.hip.cpp Makefile
	$(HIPCC) $(HIPFLAGS) --genco -o $@ kernel.hip.cpp
run: all
ifneq ($(strip $(RUN_ARGS)),)
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
else
	IRIS_ARCHS=HIP $(LAUNCHER) ./main 1024 1024 1024 1024 1000 1
	IRIS_ARCHS=HIP $(LAUNCHER) ./main 4096 4096 4096 4096 1000 0
endif
clean:
	rm -f main kernel.hip kernel.ptx *.o
.PHONY: all run clean
