CC        = g++
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

program = main
source  = main.cpp
obj     = $(source:.cpp=.o)

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall
LDFLAGS =

ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS += -g
endif
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) -p 1
