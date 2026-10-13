#include "rescue_prime.hpp"
void
prepare_mds(ulong4* const mds)
{
  for (size_t i = 0; i < STATE_WIDTH * 3; i++) {
    *(mds + i) = make_ulong4(
      MDS[i * 4 + 0], MDS[i * 4 + 1], MDS[i * 4 + 2], MDS[i * 4 + 3]);
  }
}

void
prepare_ark1(ulong4* const ark1)
{
  for (size_t i = 0; i < NUM_ROUNDS * 3; i++) {
    *(ark1 + i) = make_ulong4(
      ARK1[i * 4 + 0], ARK1[i * 4 + 1], ARK1[i * 4 + 2], ARK1[i * 4 + 3]);
  }
}

void
prepare_ark2(ulong4* const ark2)
{
  for (size_t i = 0; i < NUM_ROUNDS * 3; i++) {
    *(ark2 + i) = make_ulong4(
      ARK2[i * 4 + 0], ARK2[i * 4 + 1], ARK2[i * 4 + 2], ARK2[i * 4 + 3]);
  }
}
