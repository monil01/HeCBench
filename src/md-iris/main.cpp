#include <cassert>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <list>
#include <iostream>
#include "iris_dmem.hpp"
#include "portable_vectors.hpp"
#include "MD.h"
#include "reference.h"
#include "utils.h"

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::cout << "Usage: ./" << argv[0] << " <class size> <iteration>\n";
    return 1;
  }

  IRISSession session(&argc,&argv);
  // Problem Parameters
  int sizeClass = atoi(argv[1]);
  int iteration = atoi(argv[2]);
  #ifdef HECBENCH_BOUNDED_VALIDATION
  const int probSizes[] = {129,257,513,1025};
#else
  const int probSizes[] = { 12288, 24576, 36864, 73728 };
#endif
  assert(sizeClass >= 0 && sizeClass < 4);
  assert(iteration >= 0);

  int nAtom = probSizes[sizeClass];

  // Allocate problem data on host
  POSVECTYPE* position = (POSVECTYPE*) malloc(nAtom * sizeof(POSVECTYPE));
  FORCEVECTYPE* h_force = (FORCEVECTYPE*) malloc(nAtom * sizeof(FORCEVECTYPE));
  int *neighborList = (int*) malloc(maxNeighbors * nAtom * sizeof(int));

  std::cout << "Initializing test problem (this can take several minutes for large problems).\n";

  // Seed random number generator
  srand(123);

  // Notes on positions 
  // When the potential energy becomes exceedingly large as the distance 
  // between two atoms is very close, the host and device results may differ significantly
  for (int i = 0; i < nAtom; i++)
  {
    position[i].x = rand() % domainEdge;
    position[i].y = rand() % domainEdge;
    position[i].z = rand() % domainEdge;
  }

  std::cout << "Finished.\n";
  int totalPairs = buildNeighborList<FPTYPE, POSVECTYPE>(nAtom, position, neighborList);
  std::cout << totalPairs << " of " << nAtom*maxNeighbors << " pairs within cutoff distance = " 
       << 100.0 * ((double)totalPairs / (nAtom*maxNeighbors)) << " %\n";

  POSVECTYPE*   d_position;
  FORCEVECTYPE* d_force;
  int* d_neighborList;
  iris_alloc((void**)&d_force, nAtom * sizeof(FORCEVECTYPE));
  iris_alloc((void**)&d_position, nAtom * sizeof(POSVECTYPE));
  iris_alloc((void**)&d_neighborList, nAtom * maxNeighbors * sizeof(int));

  iris_copy(d_position, position, nAtom * sizeof(POSVECTYPE), IRISHostToDevice);
  iris_copy(d_neighborList, neighborList, nAtom * maxNeighbors * sizeof(int), IRISHostToDevice);

  dim3 grids ((nAtom+255) / 256);
  dim3 block (256);

  // warmup and result verification
  iris_launch("md", grids, block, d_position, d_force, d_neighborList,
      nAtom, maxNeighbors, lj1, lj2, cutsq);

  iris_copy(h_force, d_force, nAtom * sizeof(FORCEVECTYPE), IRISDeviceToHost);

  std::cout << "Performing Correctness Check (may take several minutes)\n";

  checkResults<FPTYPE, FORCEVECTYPE, POSVECTYPE>(h_force, position, neighborList, nAtom);

  auto start = std::chrono::steady_clock::now();

  for (int j = 0; j < iteration; j++)
  {
    iris_launch("md", grids, block, d_position, d_force, d_neighborList,
      nAtom, maxNeighbors, lj1, lj2, cutsq);
  }

  iris_sync();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time " << (time * 1e-9f) / iteration << " (s)\n";

  iris_free(d_position);
  iris_free(d_force);
  iris_free(d_neighborList);

  free(position);
  free(h_force);
  free(neighborList);

  return 0;
}
