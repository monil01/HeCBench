#pragma once
struct Point_Struct { float weight; long assign; float cost; };
static_assert(sizeof(long)==8, "CUDA source long width");
static_assert(sizeof(Point_Struct)==24, "HIP/CUDA layout");
