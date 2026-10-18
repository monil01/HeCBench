"""Independent arbitrary-precision integer lane specification and deterministic finite fixtures."""
from pathlib import Path
import json,array,hashlib
p=Path(__file__).resolve().parent;specs=json.loads((p/'operation_specs.json').read_text())['operations'];U32=(1<<32)-1

def evaluate(s,a,b,c):
    width=s['bits'];mod=1<<width;mask=mod-1;sign=1<<(width-1);result=0
    for lane in range(32//width):
        values=[(v>>(lane*width))&mask for v in (a,b,c)]
        if s['signed']:values=[v-mod if v>=sign else v for v in values]
        x,y,z=values;op=s['op'];lo=-sign if s['signed'] else 0;hi=sign-1 if s['signed'] else mask
        if op=='abs':t=abs(x)
        elif op=='abssat':t=min(abs(x),hi)
        elif op in ('absdiff','sad'):t=abs(x-y)
        elif op=='add':t=x+y
        elif op=='sub':t=x-y
        elif op=='addsat':t=min(hi,max(lo,x+y))
        elif op=='subsat':t=min(hi,max(lo,x-y))
        elif op=='avg':t=(x+y+1)//2 if x+y>=0 else (x+y)//2
        elif op=='hadd':t=(x+y)//2
        elif op=='max':t=max(x,y)
        elif op=='min':t=min(x,y)
        elif op=='max3':t=max(x,y,z)
        elif op=='min3':t=min(x,y,z)
        elif op=='neg':t=-x
        elif op=='negsat':t=min(-x,hi)
        elif op in ('addmax','addmin'):
            summed=(x+y)&mask
            if s['signed'] and summed>=sign:summed-=mod
            t=max(summed,z) if op=='addmax' else min(summed,z)
        elif op.startswith(('cmp_','set_')):
            kind,comp=op.split('_');pred={'eq':x==y,'ne':x!=y,'ge':x>=y,'gt':x>y,'le':x<=y,'lt':x<y}[comp];t=(mask if kind=='cmp' else 1) if pred else 0
        else:raise ValueError(op)
        if s['relu']:t=max(t,0)
        if op=='sad':result+=t
        else:result|=(t&mask)<<(width*lane)
    return result&U32

def signed32(v):return v-(1<<32) if v&(1<<31) else v

if __name__=='__main__':
    f=array.array('I');o=array.array('I');cases=[]
    for n in (33,1024,4097):
        a=[0x1234aba5^((-i)&U32 if i<n//2 else i) for i in range(n)];out=[]
        for i,x in enumerate(a):
            y=(x+(1 if i%2 else -1))&U32;z=x^y;assert -(1<<31)<=signed32(x)+signed32(y)<(1<<31)
            r=0
            for spec in specs:r^=evaluate(spec,x,y,z)
            out.append(r)
        f.extend((n,100));f.extend(a);o.extend(out);cases.append({'N':n,'repeats':100,'source_seed_formula':True,'signed32_add_defined_range':True,'output_count':n})
    probe_count=256;triples=array.array('I');probeoracle=array.array('I');rows=[]
    edges=[0,1,2,0x7f,0x80,0xff,0x100,0x7fff,0x8000,0xffff,0x10000,0x7fffffff,0x80000000,0xffffffff,0x7f7f7f7f,0x80808080,0x7fff7fff,0x80008000,0x00010001,0x01010101,0xffff0000,0x0000ffff]
    forced=[(0,1,0),(0xffffffff,0,0),(0x7f7f7f7f,0x01010101,0),(0x80808080,0xffffffff,0),(0x7fff7fff,0x00010001,0),(0x80008000,0xffffffff,0),(1,1,3),(0xffffffff,0xffffffff,0),(0,0xffffffff,0x80008000)]
    seed=0xc0ffee17
    for spec in specs:
        adjusted=0
        for t in range(probe_count):
            if t<len(forced):a,b,c=forced[t]
            elif t<64:a=edges[t%len(edges)];b=edges[(7*t+3)%len(edges)];c=edges[(11*t+1)%len(edges)]
            else:
                vals=[]
                for _ in range(3):seed=(1664525*seed+1013904223)&U32;vals.append(seed)
                a,b,c=vals
            if spec['signed32_add_domain_restricted'] and not (-(1<<31)<=signed32(a)+signed32(b)<(1<<31)):
                b=0;adjusted+=1
            triples.extend((a,b,c));probeoracle.append(evaluate(spec,a,b,c))
        rows.append({'id':spec['id'],'name':spec['native_name'],'probe_cases':probe_count,'signed32_add_overflow_avoided_by_declared_b_zero_fixture_policy':adjusted})
    f.extend((118,probe_count));f.extend(triples)
    (p/'fixtures_u32.bin').write_bytes(f.tobytes());(p/'oracle_u32.bin').write_bytes(o.tobytes());(p/'probe_oracle_u32.bin').write_bytes(probeoracle.tobytes())
    (p/'independent_oracle.json').write_text(json.dumps({'method':__doc__,'atol':0,'rtol':0,'cases':cases,'operations':118,'probes_per_operation':probe_count,'probe_output_count':len(probeoracle),'input_prefix_count':len(f),'probe_cases':rows,'probe_policy':'Operation-specific explicit inputtriples; packed8/16-bit extrema, wraps/saturation/averages/comparison masks and deterministic fullword values. For signed32addition only, b set0 when sum outside signed32 to avoid older-arch C++ fallback signed-overflow ambiguity. No signed32overflow generalization. Dead source predicate pointers not observable and not certified.','mapping_policy':'Native video/DPX intrinsics vs authored portable bit-lane expansion; same finite functional contract, no instruction-equivalence/performance-causality claim.','assets':{'fixture_sha256':hashlib.sha256(f.tobytes()).hexdigest(),'oracle_sha256':hashlib.sha256(o.tobytes()).hexdigest(),'probe_oracle_sha256':hashlib.sha256(probeoracle.tobytes()).hexdigest()}},indent=2)+'\n')
    print({'prefix':len(f),'main_outputs':len(o),'individual_outputs':len(probeoracle)})
