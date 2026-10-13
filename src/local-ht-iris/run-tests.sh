#!/usr/bin/env bash
set -euo pipefail
fixtures=(7-21 1-33 7-55 9-77)
for fixture in "${fixtures[@]}"; do
  kmer=${fixture##*-}
  IRIS_ARCHS=HIP ./main "../local-ht-cuda/locassm_data/localassm_extend_${fixture}.dat" "$kmer" test-out.dat
  LC_ALL=C sort test-out.dat > sorted_new
  LC_ALL=C sort "../local-ht-cuda/locassm_data/res_localassm_extend_${fixture}.dat" > sorted_res
  cmp sorted_new sorted_res
  printf 'Test for Kmer size: %s PASSED!\n' "$kmer"
done
