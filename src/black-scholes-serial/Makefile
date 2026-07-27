#===============================================================================
# Auto-generated serial Makefile (derived from -omp).
#===============================================================================

CC        = g++
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

program = main
# The Kernels and KernelsCpu .cpp files are #included by blackScholesAnalyticEngine.cpp
# (single-TU build), so we compile only that file.
source  = blackScholesAnalyticEngine.cpp
obj     = blackScholesAnalyticEngine.o

CFLAGS  := $(EXTRA_CFLAGS) -std=c++17 -Wall 

ifeq ($(DEBUG),yes)
  CFLAGS += -g
endif
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

LDFLAGS =

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 100
