"""Independent direct index permutation, including complete 128-item warp tiles."""
from pathlib import Path
import array,json,hashlib
P=Path(__file__).resolve().parent

def main():
    inputs=array.array('i');expected=array.array('i');cases=[]
    for name,n in [('source_small128',128),('source_blocktail1152',1152),('signed_bitpatterns4096',4096),('source_large1048704',1048704)]:
        a=array.array('i',range(n)) if not name.startswith('signed') else array.array('i',((i*2654435761+0x80000000)&0xffffffff for i in []))
        if name.startswith('signed'):
            for i in range(n):
                u=(i*2654435761+0x80000000)&0xffffffff;a.append(u if u<2**31 else u-2**32)
        out=array.array('i')
        for k in range(n):
            warp=k//128;lane=(k%128)//4;item=k%4;out.append(a[128*warp+32*item+lane])
        assert sorted(out)==sorted(a)
        cases.append({'name':name,'n':n,'input_offset':len(inputs),'output_offset':len(expected),'physical_blocks':(n+255)//256,'active_blocks':(n+1023)//1024,'iterations_per_active_block':1,'complete_warps':n//128,'source_adjacent_difference_checker_applicable':not name.startswith('signed')})
        inputs.extend(a);expected.extend(out);expected.extend([0x13579bdf]*16)
    (P/'fixture_i32.bin').write_bytes(inputs.tobytes());(P/'expected_i32_guards.bin').write_bytes(expected.tobytes());m={'cases':cases,'input_count':len(inputs),'output_count':len(expected),'guard_values_per_case':16,'algorithm':'Per128-itemwarptile:out[warp*128+4*lane+item]=in[warp*128+32*item+lane]','native_scope':'Full256threadblocks/4items perthread/32lanewarp,gridceil(n/256)asoriginal. Namedcases; original8192x8192x100CLIunmeasured. Inputsarefull128-itemwarptiles; arbitrarypartialwarptail isnotdefinedbyoriginaluninitializedLoadDirectBlocked','input_sha256':hashlib.sha256(inputs.tobytes()).hexdigest(),'expected_sha256':hashlib.sha256(expected.tobytes()).hexdigest(),'weak_checker_counterexample':'Onlinear-sourcecases, adding17toeverycorrectoutput preservesalladjacentdifferences=1 testedbysourcechecker but changeseveryanswer. CPUmathematicalwitnessonly.'};(P/'fixture_provenance.json').write_text(json.dumps(m,indent=2)+'\n');print({'input':len(inputs),'output':len(expected)})
if __name__=='__main__':main()
