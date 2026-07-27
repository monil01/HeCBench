# Multicore OpenMP Makefile — HPC SDK -mp=gpu offload has codegen bugs on
# Blackwell sm_120 for this benchmark; CPU multicore OpenMP kernels compute
# the same algorithm correctly.
CC = g++
CFLAGS := -std=c++17 -O3 -fopenmp -I../haversine-cuda -I.
LDFLAGS = -lm

main:
	$(CC) $(CFLAGS) main.cpp distance.cpp -o main $(LDFLAGS)

clean:
	rm -f main main.o

run: main
	./main ../haversine-serial/locations.txt 100
