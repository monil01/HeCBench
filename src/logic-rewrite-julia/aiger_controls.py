#!/usr/bin/env python3
"""Deterministic AIGER fixtures and exhaustive, independent Boolean oracle.
No CUDA/library synthesis code is used by the parser or truth evaluator.
"""
import argparse,hashlib,json,pathlib,random
PASSES=['balance','rewrite -d','refactor -m','strash','balance','rewrite -d','rewrite -z -d','rewrite -z -d','balance -s','refactor -m -z','strash','rewrite -z -d','rewrite -z -d','balance -s','strash']
def varint(n):
    b=bytearray()
    while n>127:b.append((n&127)|128);n>>=7
    b.append(n);return b

def write(path,n,edges,outs):
    data=bytearray(f'aig {n+len(edges)} {n} 0 {len(outs)} {len(edges)}\n'.encode())
    data.extend(''.join(f'{o}\n' for o in outs).encode())
    for j,(a,b) in enumerate(edges,n+1):
        a,b=max(a,b),min(a,b);lhs=2*j
        assert 0<=b<=a<lhs
        data.extend(varint(lhs-a));data.extend(varint(a-b))
    path.write_bytes(data)

def read(path):
    raw=path.read_bytes();pos=raw.index(b'\n')+1
    hdr=raw[:pos].decode().split();assert hdr[0]=='aig'
    m,n,l,o,a=map(int,hdr[1:]);assert l==0 and m==n+a
    outs=[]
    for _ in range(o):
        end=raw.index(b'\n',pos);outs.append(int(raw[pos:end]));pos=end+1
    def integer():
        nonlocal pos
        x=s=0
        while True:
            b=raw[pos];pos+=1;x|=(b&127)<<s
            if not b&128:return x
            s+=7;assert s<64
    edges=[]
    for j in range(n+1,m+1):
        x=2*j-integer();y=x-integer();assert 0<=y<=x<2*j
        edges.append((x,y))
    assert all(0<=x<=2*m+1 for x in outs)
    return n,edges,outs

def evaluate(path):
    n,edges,outs=read(path);assert n<=16
    count=1<<n;mask=(1<<count)-1
    values=[0]+[sum(((k>>j)&1)<<k for k in range(count)) for j in range(n)]
    depths=[0]*(n+1)
    def lit(x):return values[x>>1] ^ (mask if x&1 else 0)
    for a,b in edges:values.append(lit(a)&lit(b));depths.append(1+max(depths[a>>1],depths[b>>1]))
    truth=b''.join(lit(x).to_bytes((count+7)//8,'little') for x in outs)
    return dict(primary_inputs=n,outputs=len(outs),ands=len(edges),depth=max((depths[x>>1] for x in outs),default=0),assignments=count,truth_sha256=hashlib.sha256(truth).hexdigest(),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),truth

def fixture(path,n):
    rng=random.Random(20261019+n);edges=[];table={}
    def AND(a,b):
        if a==0 or b==0 or a==b^1:return 0
        if a==1:return b
        if b==1 or a==b:return a
        key=tuple(sorted((a,b)))
        if key not in table:
            edges.append(key);table[key]=2*(n+len(edges))
        return table[key]
    def OR(a,b):return AND(a^1,b^1)^1
    def XOR(a,b):return OR(AND(a,b^1),AND(a^1,b))
    pool=[2*(j+1) for j in range(n)];out=[]
    # Small structured reconvergent mux/XOR/distributive cones. Each cone
    # has <=4 distinct inputs; full resyn2 still executes all source passes.
    for k in range(32):
        a,b,c,d=[pool[(k*3+j)%n] ^ ((k>>(j+1))&1) for j in range(4)]
        if k%4==0:out.append(OR(AND(a,b),AND(a,c)))
        elif k%4==1:out.append(OR(AND(a,b),AND(a^1,c)))
        elif k%4==2:out.append(XOR(XOR(a,b),c))
        else:out.append(OR(AND(a,b),AND(c,d)))
    write(path,n,edges,out)

def main():
    p=argparse.ArgumentParser();p.add_argument('--generate',type=pathlib.Path);p.add_argument('--compare',nargs='+',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path);a=p.parse_args();rows=[]
    if a.generate:
        a.generate.mkdir(parents=True,exist_ok=True)
        for n in (4,6,9,12):
            q=a.generate/f'control_pi{n}.aig';fixture(q,n);r,t=evaluate(q);rows.append(dict(path=str(q),**r))
    if a.compare:
        base=None
        for q in a.compare:
            r,t=evaluate(q)
            if base is None:base=t
            r.update(path=str(q),full_truth_equivalent=t==base);rows.append(r)
        assert all(r['full_truth_equivalent'] for r in rows if 'full_truth_equivalent' in r),'Boolean output mismatch'
    s=json.dumps(rows,indent=2)+'\n'
    if a.output:a.output.write_text(s)
    print(s,end='')
if __name__=='__main__':main()
