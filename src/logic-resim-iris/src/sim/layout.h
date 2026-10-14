#pragma once
#include "Gate.h"
#define TRUTH_SIZE 1025
#define C_THREAD_LIMIT 32
const size_t C_PARALLEL_LIMIT = 10000;
struct gpu_GATE {
  size_t funcSer;
  Event *iPort[MAX_INPUT_PORT];
  dUnit dTable[MAX_DTABLE];
};
struct FlatGate {
  size_t funcSer;
  size_t iPort[MAX_INPUT_PORT];
  dUnit dTable[MAX_DTABLE];
};
