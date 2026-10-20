#!/usr/bin/env python3
"""Scoped shared-memory/pivot publication ordering; source math/scheduling retained."""
from pathlib import Path
import difflib,json,hashlib
P=Path(__file__).resolve().parent;s=(P/'reference_history/src/numeric.cu').read_text();old=s
warp='''        if (tidInWarp + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
        {

          subColElem = sym_c_ptr_dev[subCol] + tidInWarp + offset;
          unsigned ridx = sym_r_idx_dev[subColElem];

          if (ridx == currentCol)
          {
            s[wid] = val_dev[subColElem];
          }
          //Threads in a warp are always synchronized
          //__syncthreads();
          if (ridx > currentCol)
          {
            //elem in currentCol same row with subColElem might be 0, so
            //clearing tmpMem is necessary
            atomicAdd(&val_dev[subColElem], -tmpMem[ridx+n*bid]*s[wid]);
          }
        }
'''
new='''        const bool active = tidInWarp + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol];
        unsigned ridx = 0;
        if (active) {
          subColElem = sym_c_ptr_dev[subCol] + tidInWarp + offset;
          ridx = sym_r_idx_dev[subColElem];
          if (ridx == currentCol) s[wid] = val_dev[subColElem];
        }
        // All32 lanes of this source warp participate, including partial tails.
        __syncwarp();
        if (active && ridx > currentCol)
          atomicAdd(&val_dev[subColElem], -tmpMem[ridx+n*bid]*s[wid]);
        // Finish readers before a later warp iteration may publish another value.
        __syncwarp();
'''
assert s.count(warp)==2;s=s.replace(warp,new)
block='''    if (tid + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
    {
      subColElem = sym_c_ptr_dev[subCol] + tid + offset;
      unsigned ridx = sym_r_idx_dev[subColElem];

      if (ridx == currentCol)
      {
        s = val_dev[subColElem];
      }
      __syncthreads();
      if (ridx > currentCol)
      {
        atomicAdd(&val_dev[subColElem], -tmpMem[stream * n + ridx] * s);
      }
    }
'''
newblock='''    const bool active = tid + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol];
    unsigned ridx = 0;
    if (active) {
      subColElem = sym_c_ptr_dev[subCol] + tid + offset;
      ridx = sym_r_idx_dev[subColElem];
      if (ridx == currentCol) s = val_dev[subColElem];
    }
    __syncthreads();
    if (active && ridx > currentCol)
      atomicAdd(&val_dev[subColElem], -tmpMem[stream * n + ridx] * s);
    __syncthreads();
'''
assert s.count(block)==1;s=s.replace(block,newblock)
# Move redundant same-value global pivot publication to one thread before readers.
# Preserve original no-L-entry case: it never clamps its pivot.
for name in ['RL_perturb','RL_onecol_factorizeCurrentCol_perturb']:
 start=s.index('__global__ void '+name+'(');end=s.find('__global__ void ',start+1)
 if end<0:end=s.index('void LUonDevice(',start)
 part=s[start:end]
 pivot='''      if (abs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
        val_dev[l_col_ptr_dev[currentCol]] = pert;

'''
 assert part.count(pivot)==1;part=part.replace(pivot,'')
 marker='  //update current col\n'
 insertion='''  if (tid == 0 && currentLColSize > 0 && abs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
    val_dev[l_col_ptr_dev[currentCol]] = pert;
  __syncthreads();

'''
 assert part.count(marker)==1;part=part.replace(marker,insertion+marker)
 s=s[:start]+part+s[end:]
(P/'corrected_numeric.cuh').write_text(s)
(P/'numeric_ordering.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='reference_history/src/numeric.cu',tofile='corrected_numeric.cuh')))
(P/'ordering_correction_provenance.json').write_text(json.dumps(dict(native_reference_repaired=True,intervention='Explicitsource sharedpublish/use/reuse ordering onVolta; singlethreadsame-positive-pert pivotpublication onlywhenLnonempty. Sparseupdate/rule/levelwork andstorage unchanged.',original_sha256=hashlib.sha256(old.encode()).hexdigest(),corrected_sha256=hashlib.sha256(s.encode()).hexdigest(),witness='shared_ordering_witness.json',raw_warning_log='.porting-state/porting_logs/slu-julia-shared/original_add32_racecheck.log',numeric_wrong_output_observed_in_first_unmodified_CLI=False,source_ABFT_passed_despite_shared_hazard_warnings=True,additional_onecol_shared_reuse_and_perturb_ordering_basis='Staticdata-dependency/lifetime observations; notclaimedobservedGPUwrongoutput.',exclusion='Conditionalconcurrency-repairedreference; exclude untouched-native/performanceaggregates. No frameworkunsupportedclaim.'),indent=2)+'\n')
