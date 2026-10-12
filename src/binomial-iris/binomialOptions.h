#pragma once

#include "realtype.h"

typedef struct
{
    real S;
    real X;
    real T;
    real R;
    real V;
} TOptionData;

#define NUM_STEPS 2048
#define MAX_OPTIONS 1024
#define NUM_ITERATIONS 1000
