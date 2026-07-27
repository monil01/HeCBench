#===============================================================================
# Auto-generated serial Makefile (derived from -omp).
#===============================================================================

CC        = g++
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

program = main
source  = Timer.cpp main.cpp OptionParser.cpp triad.cpp Option.cpp
obj     = Timer.o main.o OptionParser.o triad.o Option.o

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
	$(LAUNCHER) ./$(program) --passes 100 -v
