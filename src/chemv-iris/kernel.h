#ifndef CHEMV_IRIS_KERNEL_H
#define CHEMV_IRIS_KERNEL_H

struct ComplexFloat {
  float Re;
  float Im;
};

#define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
#define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })

#endif
