"""CPU-created original DCN fixtures, GPU DMem computation, CPU-only oracle."""
import argparse, math, os, subprocess, tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
parser=argparse.ArgumentParser()
parser.add_argument('--small',action='store_true')
parser.add_argument('--offset-fixture',action='store_true',help='additional nonzero-offset validation, not production initialization')
args=parser.parse_args()
B,C,O,H,W,G=(2,4,3,5,7,2) if args.small else (100,64,64,128,128,2)
torch.manual_seed(0)
x=torch.randn(B,C,H,W,requires_grad=True)
w=torch.empty(O,C,3,3).uniform_(-1/math.sqrt(C*9),1/math.sqrt(C*9)).requires_grad_()
bias=torch.zeros(O,requires_grad=True)
# Constructing original nn.Conv2d consumes these RNG draws, even though zeroed.
module=torch.nn.Conv2d(C,G*27,3,padding=1)
pw=module.weight.detach().zero_().requires_grad_();pb=module.bias.detach().zero_().requires_grad_()
if args.offset_fixture:
    with torch.no_grad():pw.uniform_(-0.015,0.015);pb.uniform_(-0.1,0.1)
# Original target is GPU-generated; CPU uniform substitutes explicitly because
# the local torch build has no HIP support. Loss derivatives are target-independent.
target=torch.empty(B,O,H,W).uniform_(-0.01,0.01)
root=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='iris-dcn-') as tmp:
    p=str(Path(tmp)/'tensor')
    for name,a in [('x',x),('w',w),('bias',bias),('pw',pw),('pb',pb),('target',target)]:a.detach().numpy().tofile(p+'.'+name)
    subprocess.run([str(root/'main'),*map(str,(B,C,O,H,W,G)),p],cwd=root,env={**os.environ,'IRIS_ARCHS':'HIP'},check=True)
    # Independent differentiable bilinear gather rather than source im2col code.
    pred=F.conv2d(x,pw,pb,padding=1);pred.retain_grad()
    yy,xx=torch.meshgrid(torch.arange(H),torch.arange(W),indexing='ij')
    columns=[]
    for c in range(C):
        group=c//(C//G)
        plane=x[:,c].reshape(B,-1)
        for k in range(9):
            h=yy+k//3-1+pred[:,group*18+2*k]
            a=xx+k%3-1+pred[:,group*18+2*k+1]
            iy=h.floor();ix=a.floor();fy=h-iy;fx=a-ix
            val=0
            for dy in range(2):
                for dx in range(2):
                    y=iy+dy;z=ix+dx
                    valid=(y>=0)&(y<H)&(z>=0)&(z<W)
                    gathered=plane.gather(1,(y.clamp(0,H-1)*W+z.clamp(0,W-1)).long().reshape(B,-1)).reshape(B,H,W)
                    val=val+gathered*valid*(fy if dy else 1-fy)*(fx if dx else 1-fx)
            inside=(h>-1)&(h<H)&(a>-1)&(a<W)
            columns.append(val*inside*torch.sigmoid(pred[:,G*18+group*9+k]))
    col=torch.stack(columns,1)
    out=torch.einsum('oc,bchw->bohw',w.reshape(O,-1),col)+bias[None,:,None,None]
    error=(target-out).mean();error.backward()
    tensors=[('out',out),('mean',error.reshape(1)),('gx',x.grad),('gw',w.grad),('gb',bias.grad),('gp',pred.grad),('gpw',pw.grad),('gpb',pb.grad)]
    for name,a in tensors:
        expected=a.detach().numpy();got=np.fromfile(p+'.'+name,dtype=np.float32).reshape(expected.shape)
        np.testing.assert_allclose(got,expected,rtol=5e-5,atol=2e-6)
        print(f'PASS {name} outputs={got.size} max_abs={np.max(np.abs(got-expected))}',flush=True)
    print('PASS complete DCN forward/backward shape=',(B,C,O,H,W,G),'nonzero_offsets=',args.offset_fixture,flush=True)
