#!/usr/bin/env python3
# Exact CPU transcription of source warpBitonicSort comparison/selection network.
import struct,json,collections
from pathlib import Path
P=Path(__file__).resolve().parent;c=json.loads((P/'cases.json').read_text())[-2];a=c['bits']+[0xff800000]*18
fl=lambda b:struct.unpack('<f',struct.pack('<I',b))[0]
for mask,bit in [(1,1),(2,2),(1,2),(4,3),(2,3),(1,3),(8,4),(4,4),(2,4),(1,4),(16,None),(8,None),(4,None),(2,None),(1,None)]:
 old=a[:]
 for lane in range(32):
  direction=((lane>>bit)&1)^((lane>>(mask.bit_length()-1))&1)if bit is not None else((lane>>(mask.bit_length()-1))&1)
  y=old[lane^mask];a[lane]=y if int(fl(old[lane])>fl(y))==direction else old[lane]
print(json.dumps({'input_bits':c['bits'],'source_network_output_bits':a[:14],'input_zero_sign_counts':dict(collections.Counter(hex(v)for v in c['bits']if fl(v)==0)),'output_zero_sign_counts':dict(collections.Counter(hex(v)for v in a[:14]if fl(v)==0)),'numeric_ordered':all(fl(a[i])>=fl(a[i+1])for i in range(13)),'interpretation':'Source comparator only numeric>, so signedzeros equivalent; bit-multiset is stronger than source numeric contract. No numerical sorting defect inferred.'},indent=2))
