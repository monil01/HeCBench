# Multicore OpenMP Makefile — HPC SDK -mp=gpu offload has codegen bugs on
# Blackwell sm_120 for this benchmark; CPU multicore OpenMP kernels compute
# the same algorithm correctly.
CC = g++
CFLAGS := -std=c++17 -O3 -fopenmp  -I.
LDFLAGS = -lm

main:
	$(CC) $(CFLAGS) main.cpp Option.cpp OptionParser.cpp Timer.cpp triad.cpp -o main $(LDFLAGS)

clean:
	rm -f main main.o

run: main
	./main --passes 100 -v
