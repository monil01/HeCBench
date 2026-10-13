#pragma once
struct RFResult { double x,y,z; };
static_assert(sizeof(RFResult)==3*sizeof(double));
