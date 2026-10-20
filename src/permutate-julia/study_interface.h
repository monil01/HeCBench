#pragma once
#include <cstdint>
#include <cstddef>
struct PLaunch{uint32_t size,len,N,blocks,threads,blen;double mean,median;void*pointers[6];};
static_assert(sizeof(PLaunch)==88&&offsetof(PLaunch,pointers)==40,"PLaunch ABI");
using PCallback=int(*)(const PLaunch*);
