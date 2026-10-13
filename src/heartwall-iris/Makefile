IRIS_ROOT ?= /tmp/iris
IRIS_BUILD ?= $(IRIS_ROOT)/build
CXX ?= g++
CC ?= gcc
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -Wall
CFLAGS ?= -O3
CPPFLAGS += -I$(IRIS_ROOT)/include -I$(IRIS_BUILD)/include -Iutil/file -DOUTPUT
LDFLAGS += -L$(IRIS_BUILD)/src/runtime -Wl,-rpath,$(IRIS_BUILD)/src/runtime
LDLIBS += -liris -ldl -lpthread -lm
RUN_ARGS ?= 104
LAUNCHER ?=
UTILS = util/avi/avilib.o util/avi/avimod.o util/file/file.o
all: main kernel.hip
main: main.cpp wrapper.cpp main.h $(UTILS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) main.cpp wrapper.cpp $(UTILS) -o $@ $(LDFLAGS) $(LDLIBS)
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
kernel.hip: kernel.hip.cpp main.h
	$(HIPCC) -O3 --genco kernel.hip.cpp -o $@
run: all
	IRIS_ARCHS=HIP $(LAUNCHER) ./main $(RUN_ARGS)
clean:
	rm -f main kernel.hip $(UTILS)
.PHONY: all run clean
