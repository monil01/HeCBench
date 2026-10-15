"""Original test suite with CPU-only tensor creation and independent oracles."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--small', action='store_true', help='bounded validation, not the production shapes')
args = parser.parse_args()
root = Path(__file__).resolve().parent

def cdist(a, b, metric='braycurtis'):
    # Row-wise long-double oracle bounds temporary storage at production shapes.
    aa=a.astype(np.longdouble); bb=b.astype(np.longdouble)
    out=np.empty((len(a),len(b)),dtype=np.longdouble)
    for i, row in enumerate(aa):
        out[i]=np.abs(row-bb).sum(axis=1)/np.abs(row+bb).sum(axis=1)
    return out

def case(B, M, N, D, dtype, gradient=False):
    torch.manual_seed(0)
    # Forward tests in the CUDA source also create these tensors on the CPU.
    x = torch.rand(B, M, D, dtype=dtype).numpy()
    y = torch.rand(B, N, D, dtype=dtype).numpy()
    with tempfile.TemporaryDirectory(prefix='iris-braycurtis-') as tmp:
        p = Path(tmp) / 'tensor'
        x.tofile(str(p)+'.x'); y.tofile(str(p)+'.y')
        g = np.linspace(-0.8, 1.3, B*M*N).reshape(B,M,N).astype(x.dtype)
        if gradient: g.tofile(str(p)+'.g')
        subprocess.run([str(root/'main'), 'float' if dtype==torch.float32 else 'double', str(B),str(M),str(N),str(D),str(p),str(int(gradient))],cwd=root,env={**os.environ,'IRIS_ARCHS':'HIP'},check=True)
        out = np.fromfile(str(p)+'.out',dtype=x.dtype).reshape(B,M,N)
        reference = np.stack([cdist(a,b,'braycurtis') for a,b in zip(x,y)])
        np.testing.assert_allclose(out,reference,rtol=2e-6 if dtype==torch.float32 else 2e-13,atol=2e-7 if dtype==torch.float32 else 2e-14)
        print(f'PASS forward dtype={dtype} shape={B,M,N,D} outputs={out.size} max_abs={np.max(np.abs(out-reference))}',flush=True)
        if gradient:
            for name, arr in [('gx',x),('gy',y)]:
                got=np.fromfile(str(p)+'.'+name,dtype=x.dtype).reshape(arr.shape)
                ref=np.empty_like(arr)
                for index in np.ndindex(arr.shape):
                    original=arr[index]; eps=1e-6
                    arr[index]=original+eps
                    hi=sum(np.sum(cdist(a,b,'braycurtis')*gg) for a,b,gg in zip(x,y,g))
                    arr[index]=original-eps
                    lo=sum(np.sum(cdist(a,b,'braycurtis')*gg) for a,b,gg in zip(x,y,g))
                    arr[index]=original; ref[index]=(hi-lo)/(2*eps)
                np.testing.assert_allclose(got,ref,rtol=1e-5,atol=1e-7)
                print(f'PASS finite-difference {name} outputs={got.size} max_abs={np.max(np.abs(got-ref))}',flush=True)

case(1,2,2,2,torch.float64,True)
case(2,1,1,2,torch.float64,True)
for dtype in [torch.float32,torch.float64]:
    case(1,17,29,31,dtype) if args.small else case(1,2000,5000,768,dtype)
    case(3,11,13,19,dtype) if args.small else case(8,1000,2500,768,dtype)
