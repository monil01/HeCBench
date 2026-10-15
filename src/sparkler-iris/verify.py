#!/usr/bin/env python3
"""Check every rank/iteration/active-step cell and original reduced checksum."""
import array
import math
import pathlib
import re
import struct
import sys

def read(prefix, rank):
    records = {}
    with pathlib.Path(f'{prefix}.rank{rank}.bin').open('rb') as f:
        while header := f.read(64):
            assert len(header) == 64
            r,iteration,step,left,right,m,n,k = struct.unpack('<8Q',header)
            assert r == rank and (iteration,step) not in records
            data = f.read(m*n*4)
            assert len(data) == m*n*4
            values = array.array('f'); values.frombytes(data)
            records[iteration,step] = (left,right,m,n,k,values)
    return records

def verify(native, iris, native_log, iris_log, ranks, vectors, fields, iterations):
    count, checksum, blocks = 0,0,0
    m = 2*((vectors+ranks-1)//ranks)
    for rank in range(ranks):
        a,b = read(native,rank),read(iris,rank)
        steps=(ranks+2)//2-(1 if ranks%2==0 and rank>=ranks//2 else 0)
        expected_keys={(i,s) for i in range(1,iterations+1) for s in range(1,steps+1)}
        assert a.keys()==b.keys()==expected_keys
        left=vectors*rank//ranks
        local=vectors*(rank+1)//ranks-left
        for iteration,step in sorted(expected_keys):
            other=(rank+step-1)%ranks
            right=vectors*other//ranks
            localright=vectors*(other+1)//ranks-right
            x,y=a[iteration,step],b[iteration,step]
            assert x[:5]==y[:5]==(left,right,m,m,fields)
            assert x[5].tobytes()==y[5].tobytes(),f'GPU mismatch rank {rank}, step {step}'
            for col in range(m):
                for row in range(m):
                    stride=math.lcm(1+(left+row)%499,1+(right+col)%499)
                    expected=1+(fields-1)//stride
                    assert x[5][row+m*col]==expected,(rank,iteration,step,row,col)
                    if row<local and col<localright and not(step==1 and row>=col):
                        checksum+=1+(expected*(row+left)*(col+right))%(1<<40)
            count+=m*m; blocks+=1
    checksum%=1<<64
    for log in [native_log,iris_log]:
        hashes=re.findall(r'^hash (\d+)$',pathlib.Path(log).read_text(),re.M)
        assert hashes==[str(checksum)],(log,hashes,checksum)
    print(f'PASS ranks={ranks} vectors={vectors} fields={fields} iterations={iterations}: all {count} matrix values in {blocks} active blocks bit-exact native/IRIS and exact independent LCM-count oracle; MPI hash={checksum}')

if __name__=='__main__':
    verify(*sys.argv[1:5],*map(int,sys.argv[5:]))
