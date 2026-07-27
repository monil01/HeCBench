CC = nvcc
OPTIMIZE = yes
LAUNCHER =
ARCH = sm_120

program = main
CFLAGS := -std=c++17 -Xcompiler -Wall -arch=$(ARCH)

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

$(program): main.cu
	$(CC) $(CFLAGS) main.cu -o $(program)

clean:
	rm -rf $(program) *.o

run: $(program)
	$(LAUNCHER) ./$(program) 4194304 100
