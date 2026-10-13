#pragma once
static const int MaxPins = 256;
using ID = short;
using coord_t = int;
struct edge { ID src; ID dst; };
static_assert(sizeof(edge) == 4, "edge ABI");
