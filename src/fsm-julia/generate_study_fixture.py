"""Independent sparse-state CPU genetic-search oracle and source rand fixtures.
No CUDA or Julia implementation is imported. The low-three-bit recurrence proves
that all 1024 islands share observable evolution; full RNG seeds use affine jumps.
"""
from pathlib import Path
import ctypes,json,struct,hashlib
P=Path(__file__).resolve().parent
MASK=(1<<31)-1; A=26757677

def step(x):return (A*x+1)&MASK

def jump(x,n):
    a,c=1,0; b,d=A,1
    while n:
        if n&1:a,c=(b*a)&MASK,(b*c+d)&MASK
        b,d=(b*b)&MASK,(b*d+d)&MASK;n>>=1
    return (a*x+c)&MASK

def score(machine,trace):
    table={};hits=0
    for d in trace:
        pc=d>>1;s=table.get(pc,0);bit=d&1
        hits+=int((s&1)==bit);table[pc]=machine[2*s+bit]
    return hits

def island(trace,block):
    rng=[step(1234^(256*block+t)) for t in range(256)]
    fsms=[]
    for t in range(256):
        f=[]
        for i in range(16):rng[t]=step(rng[t]);f.append(rng[t]&7)
        fsms.append(f)
    old=0;same=0;generations=0;wins=[0]*256;history=[]
    while same<1:
        fitness=[score(f,trace) for f in fsms]
        best=max(range(256),key=lambda t:(fitness[t],t));top=fitness[best]
        wins[best]+=1;generations+=1;same+=1
        if top>old:old=top;same=0
        history.append({'generation':generations,'fitness':top,'winner_lane':best})
        winner=fsms[best][:]
        for t in range(256):
            mutate=False
            if t!=best:rng[t]=step(rng[t]);mutate=(rng[t]&7)==0
            for i in range(16):
                rng[t]=step(rng[t]);u=rng[t];rng[t]=step(rng[t]);u &=rng[t]
                if t!=best:fsms[t][i]=((winner[i]^u)&7) if mutate else ((fsms[t][i]&u)|(winner[i]&~u))
        if generations>10000:raise RuntimeError('oracle nonconvergence')
    for t in range(256):assert rng[t]==jump(1234^(256*block+t),17+33*generations-wins[t])
    assert score(fsms[best],trace)==top
    return {'machine':fsms[best],'smax':top,'sbest':best,'oldmax':old,'same':same,'generations':generations,'wins':wins,'history':history}

def main():
    libc=ctypes.CDLL(None);cases=[];flatinputs=[];flatoutputs=[]
    for name,n in [('rand257',257),('rand4099',4099),('repeated_pc1021',1021)]:
        libc.srand(123)
        trace=[libc.rand()&65535 for _ in range(n)] if name.startswith('rand') else [2*((i//13)%17)+((i//3+i//17)&1) for i in range(n)]
        result=island(trace,0);last=island(trace,1023);assert result==last
        g=result['generations'];wins=result['wins'];machine=result['machine']
        best=[1023,result['smax'],1024*g]+machine
        rng=[jump(1234^gid,17+33*g-wins[gid%256]) for gid in range(262144)]
        output=best+rng+machine*1024+[result['same']]*1024+[result['smax']]*1024+[result['sbest']]*1024+[result['oldmax']]*1024
        flatinputs+=trace;flatoutputs+=output
        cases.append({'name':name,'length':n,'input_count':len(trace),'output_count':len(output),'oracle':result,'fitness_confirmed_independently':score(machine,trace),'distinct_pc':len({d>>1 for d in trace})})
    for fn,values,fmt in [('fixture_u16.bin',flatinputs,'H'),('expected_u64.bin',flatoutputs,'Q')]:
        data=struct.pack('<'+fmt*len(values),*values);(P/fn).write_bytes(data)
    meta={'cases':cases,'input_total':len(flatinputs),'output_total':len(flatoutputs),'population_blocks':1024,'population_threads':256,'table_bytes_per_thread':32768,'source_rng_seed':1234,'source_trace_rand_seed':123,'oracle':'independent sparse per-PC state and explicit simultaneous generation; affine LCG jump all RNG states; islands 0 and1023 independently simulated','island_equivalence_proof':'LCG modulo8 uses a=5,c=1; all initialization,mutation,crossover depend only on low3bits. block offset256 does not change seed low3bits. Corresponding lanes therefore evolve identically by induction. Highbits preserved by jump verification.','input_sha256':hashlib.sha256((P/'fixture_u16.bin').read_bytes()).hexdigest(),'expected_sha256':hashlib.sha256((P/'expected_u64.bin').read_bytes()).hexdigest(),'scope':'Full source1024x256 population,32768-byte state,8-state FSM,cutoff1,repeat10; named shorter traces. Original65536 CLI unmeasured. Training-trace fitness only, no heldout prediction claim.'}
    (P/'fixture_provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    print([(c['name'],c['oracle']['generations'],c['oracle']['smax'],c['oracle']['sbest'])for c in cases])
if __name__=='__main__':main()
