#===============================================================================
# Auto-generated serial Makefile (derived from -omp).
#===============================================================================

CC        = g++
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

program = main
source  = bfs.cpp
obj     = bfs.o

CFLAGS  := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../bfs-sycl

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
	$(LAUNCHER) ./$(program) ../data/bfs/graph1MW_6.txt
