import argparse,os,subprocess,tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--small',action='store_true');p.add_argument('--T',type=int);p.add_argument('--native',type=Path,help='optional standalone harness of unchanged native HIP kernels');a=p.parse_args()
B,C,T=(8,3,16) if a.small else (32,768,768)
if a.T:T=a.T
# Original sequence: two complete warmups then one measured computation.
repeat = 1 if a.small else 3

def case():
    torch.manual_seed(42)
    # Explicit CPU RNG substitution for unavailable original HIP PyTorch generator.
    w=torch.rand(C,T,requires_grad=True);k=torch.rand(B,C,T,requires_grad=True)
    root=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='iris-timex-') as tmp:
        prefix=str(Path(tmp)/'t');w.detach().numpy().tofile(prefix+'.w');k.detach().numpy().tofile(prefix+'.k')
        subprocess.run(['/usr/bin/time','-f','iris_compute_seconds %e',str(root/'main'),str(B),str(C),str(T),'0.1',prefix],cwd=root,env={**os.environ,'IRIS_ARCHS':'HIP'},check=True)
        out=F.conv1d(F.pad(k,(T-1,0)),w.unsqueeze(1),groups=C)+0.1
        loss=(out.square()-out.tanh()).sum();loss.backward()
        for name,expected in [('out',out.detach().numpy()),('gw',w.grad.numpy()),('gk',k.grad.numpy()),('loss',loss.detach().numpy().reshape(1))]:
            got=np.fromfile(prefix+'.'+name,dtype=np.float32).reshape(expected.shape)
            np.testing.assert_allclose(got,expected,rtol=1e-5,atol=1e-5)
            print(f'PASS CPU-conv1d {name} outputs={got.size} max_abs={np.max(np.abs(got-expected))}',flush=True)
        if a.native:
            subprocess.run(['/usr/bin/time','-f','native_runtime_seconds %e',str(a.native.resolve()),str(B),str(C),str(T),prefix],check=True)
            for name in ['out','batchgw','gk']:
                native=np.fromfile(prefix+'.native_'+name,dtype=np.float32);iris=np.fromfile(prefix+'.'+name,dtype=np.float32)
                np.testing.assert_array_equal(iris,native)
                print(f'PASS original-HIP-kernel {name} outputs={iris.size} exact',flush=True)
        print('PASS complete TimeX forward, nonlinear loss, backward and batch reduction',B,C,T,flush=True)
for iteration in range(repeat):
    case()
