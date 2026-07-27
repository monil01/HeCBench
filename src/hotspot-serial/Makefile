#===============================================================================
# Auto-generated serial Makefile (derived from -omp).
#===============================================================================

CC        = g++
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

program = main
source  = hotspot.cpp
obj     = hotspot.o

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
	$(LAUNCHER) ./$(program) 512 2 200 ../data/hotspot/temp_512 ../data/hotspot/power_512 output_512.out
