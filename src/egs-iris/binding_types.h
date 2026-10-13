#pragma once
struct source_bind_t {uint n;float px,py,pz,z,minx,miny,maxx,maxy,sizex,sizey,area;};
struct detector_bind_t {float x,y,z,dx,dy;uint nx,ny;};
struct phantom_bind_t {uint nx,ny,nz;};
static_assert(sizeof(source_bind_t)==48,"source binding ABI");
static_assert(sizeof(detector_bind_t)==28,"detector binding ABI");
static_assert(sizeof(phantom_bind_t)==12,"phantom binding ABI");
