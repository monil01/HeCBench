#ifndef BTREE_IRIS_WRAPPERS_H
#define BTREE_IRIS_WRAPPERS_H

#include "./common.h"

void kernel_wrapper(record *records,
                    long records_mem,
                    knode *knodes,
                    long knodes_elem,
                    long knodes_mem,
                    int order,
                    long maxheight,
                    int count,
                    long *currKnode,
                    long *offset,
                    int *keys,
                    record *ans);

void kernel2_wrapper(knode *knodes,
                     long knodes_elem,
                     long knodes_mem,
                     int order,
                     long maxheight,
                     int count,
                     long *currKnode,
                     long *offset,
                     long *lastKnode,
                     long *offset_2,
                     int *start,
                     int *end,
                     int *recstart,
                     int *reclength);

#endif
