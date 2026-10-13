IRIS_ROOT ?= /tmp/iris
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -I$(IRIS_ROOT)/include
LDFLAGS ?= -L$(IRIS_ROOT)/build/src/runtime -Wl,-rpath,$(IRIS_ROOT)/build/src/runtime -liris -lpthread -ldl
RUN_ARGS ?= file ../data/b+tree/mil.txt command ../data/b+tree/command.txt

all: main kernel.hip

main: main.cpp iris_wrappers.cpp util/num/num.c
	$(CXX) $(CXXFLAGS) -I. -o $@ main.cpp iris_wrappers.cpp util/num/num.c $(LDFLAGS)

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $<

run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)

clean:
	rm -f main kernel.hip *.o util/num/*.o output.txt
