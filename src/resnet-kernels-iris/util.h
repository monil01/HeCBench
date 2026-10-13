#ifndef __UTIL_H__
#define __UTIL_H__

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iris/iris.hpp>
#include <vector>
#include <initializer_list>
void launch(const char*, std::initializer_list<size_t>, std::initializer_list<size_t>,
            std::initializer_list<iris::DMem*>);
void dump_result(const float*, int);

float* get_parameter(const char* filename, int size);

float* transpose(float* weight, int h, int w);

void output_checker(float* A, float* B, int len, int channel, int shift);

#endif

