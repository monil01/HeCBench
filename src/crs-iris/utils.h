#ifndef __UTILS
#define __UTILS

#include <stdio.h>
#include <stdlib.h>
#include <chrono>

size_t align_value(size_t valueToAlign, size_t alignMask);

void generateRandomValue(char *data, size_t size);
typedef const char *coding_func;


#define talloc(type, num) (type *)malloc(sizeof(type) * (num))

#define MIN_K 1
#define MAX_K 4

#define MIN_M 1
#define MAX_M 4

#define MIN_W 4
#define MAX_W 8

#define MAX_K_MULIPLY_M (MAX_K * MAX_M)

#define WARM_UP_SIZE 4

#define MAX_THREAD_NUM 128
#define THREADS_PER_BLOCK 128
#define WARP_SIZE 32

#endif

