"""Source-index proof, no device execution or inferred scientific repair history."""
import json
from pathlib import Path
reads=[(x,y,r,c,(4*x+r)*16+4*y+c) for x in range(4) for y in range(4) for r in range(6) for c in range(6)]
bad=[x for x in reads if x[-1]>=256]
assert max(x[-1] for x in reads)==289 and bad
for channels in (128,256):
 assert max(x[-1]*channels+channels-1 for x in reads)==290*channels-1
print(json.dumps({'source_expression':'(blockIdx.x*4+i)*16+blockIdx.y*4+threadIdx.y','read_vectors':len(reads),'out_of_original_allocation_vector_reads':len(bad),'maximum_channel_vector':289,'minimum_required_physical_channel_vectors':290,'tail_extension_channel_vectors':34,'first_invalid':bad[0],'last_invalid':bad[-1],'contract':'Original256vectors retained;34explicitzeros makeallloadsdefined. Kernelunchanged. No original16x16allocationCLI passclaim.'},indent=2))
