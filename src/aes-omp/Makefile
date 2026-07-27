# Multicore OpenMP Makefile — fresh SBOX+XOR kernel (matches aes-triton
# simplification). HPC SDK -mp=gpu on Blackwell fails the AES round math.
CC = gcc
CFLAGS := -O3 -fopenmp

main:
	$(CC) $(CFLAGS) main_mc.c -o main -lm

clean:
	rm -f main main_mc

run: main
	./main 100 0 ../urng-sycl/URNG_Input.bmp
