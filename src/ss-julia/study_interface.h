#pragma once
#include <cstdint>
struct SSCall{int op,threads,blocks,smem;uint32_t text_length,pattern_length,max_search,pad;void*pointers[4];};
static_assert(sizeof(SSCall)==64,"SSCallfixedABI");using SSCallback=int(*)(SSCall*);
