#!/usr/bin/env python3
"""Independent complete suffix-array certificate, not a DC3 implementation.
After permutation validation, build inverse ranks. Check that adjacent output
suffixes are strictly ordered by (first input symbol, inverse rank of tail),
with empty tail rank -1. This suffices: a shortest-tail counterexample to full
lexicographic order would reduce to a shorter counterexample; the empty-tail
base case is fixed. Thus the complete unique suffix order is certified in O(n).
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input-i32',type=Path,required=True);p.add_argument('--suffix-i32',type=Path,required=True);p.add_argument('--n',type=int,required=True);p.add_argument('--source-bytes',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();s=np.fromfile(a.input_i32,dtype='<i4');out=np.fromfile(a.suffix_i32,dtype='<i4');n=a.n;assert len(s)==n+3 and len(out)==n+3;assert np.all(s[n:]==0)and np.all(out[n:]==0);sa=out[:n];assert np.all((0<=sa)&(sa<n));assert np.all(np.bincount(sa,minlength=n)==1);rank=np.empty(n+1,dtype='<i4');rank[n]=-1;rank[sa]=np.arange(n,dtype='<i4');characters=s[sa];tails=rank[sa+1];ordered=(characters[:-1]<characters[1:])|((characters[:-1]==characters[1:])&(tails[:-1]<tails[1:]));assert np.all(ordered),('lexicographic_violation_positions',np.flatnonzero(~ordered)[:10].tolist())
source_exact=None
if a.source_bytes:
 raw=a.source_bytes.read_bytes()[:n];assert len(raw)==n;assert np.array_equal(s[:n],np.frombuffer(raw,dtype='i1').astype('<i4'));source_exact=True
r={'status':'pass','n':n,'input_i32':str(a.input_i32),'input_sha256':hashlib.sha256(a.input_i32.read_bytes()).hexdigest(),'suffix_i32':str(a.suffix_i32),'suffix_sha256':hashlib.sha256(a.suffix_i32.read_bytes()).hexdigest(),'all_n_positions_are_permutation':True,'all_n_minus_one_lexicographic_key_checks':int(ordered.size),'padding_three_words_exact_zero':True,'actual_GPU_inputs_match_original_source_bytes':source_exact,'certificate':'fullpermutation+strict(first_symbol,inverse_rank_of_tail)order; uniquecompletelexicographicsuffixorderbyfinite-tailinduction, notchecksum','CPU_expected_suffix_array_generated':False,'NumPy_version':np.__version__,'scope':'AllnactualGPUoutputindices checked, notsampled; corpusASCIIpositiveandnativezero-sentinelpolicy explicit'};a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
