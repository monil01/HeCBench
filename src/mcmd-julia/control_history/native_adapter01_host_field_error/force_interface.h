#pragma once
// Exact original d_atom ABI is112bytes, forceoffset48,molidoffset72.
using MCMDCallback=int(*)(double*,int,int,int,double,double,int,int,double,const double*,const double*);
extern MCMDCallback mcmd_callback;
extern double mcmd_excluded_seconds;
void mcmd_capture_input(const void*,int,int,int,double,double,int,int,double,const double*,const double*);
void mcmd_capture_output(const void*,const void*,int,int,double,double,int,int,const double*,const double*);
