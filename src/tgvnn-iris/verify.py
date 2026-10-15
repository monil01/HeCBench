#!/usr/bin/env python3
"""Independent complex source oracle; CPU validation only, never benchmark fallback."""
import argparse
import math
import pathlib
import re
import struct
import numpy as np

def ra(path):
    with pathlib.Path(path).open('rb') as f:
        magic,flags,kind,width,size,nd=struct.unpack('<6Q',f.read(48))
        dims=struct.unpack('<'+'Q'*nd,f.read(8*nd))
        raw=f.read(size)
    assert magic==0x7961727261776172 and flags==0 and kind==4 and width==8
    assert size==8*math.prod(dims) and len(raw)==size
    return dims,np.frombuffer(raw,dtype=np.complex64).copy()

def compare(label,actual,expected,atol=3e-5,rtol=2e-3,l2_limit=3e-4):
    assert actual.shape==expected.shape,(label,actual.shape,expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(),label
    error=np.abs(actual-expected)
    limit=atol+rtol*np.abs(expected)
    bad=np.count_nonzero(error>limit)
    relative=np.linalg.norm((actual-expected).ravel().astype(np.complex128))/max(np.linalg.norm(expected.ravel().astype(np.complex128)),1e-30)
    print(f'{label}: count={actual.size} max_abs={error.max():.9g} relative_L2={relative:.9g} bad={bad}',flush=True)
    assert bad==0 and relative<l2_limit,label

def grad(x,axis,mu):
    out=np.zeros_like(x)
    a=[slice(None)]*3;b=a.copy();a[axis]=slice(None,-1);b[axis]=slice(1,None)
    out[tuple(a)]=x[tuple(b)]-x[tuple(a)]
    if axis==0:out/=np.float32(mu)
    return out

def adj(x,axis,mu):
    out=np.zeros_like(x)
    first=[slice(None)]*3;first[axis]=0
    last=first.copy();last[axis]=-1
    penult=first.copy();penult[axis]=-2
    a=first.copy();a[axis]=slice(1,-1)
    b=first.copy();b[axis]=slice(None,-2)
    out[tuple(first)]=x[tuple(first)]
    out[tuple(last)]=-x[tuple(penult)]
    out[tuple(a)]=x[tuple(a)]-x[tuple(b)]
    # Faithfully retain source's temporal boundary convention: only interior
    # differences divide by mu, not the two temporal boundary assignments.
    if axis==0:out[tuple(a)]/=np.float32(mu)
    return out

def proximal(x,beta):
    nt,cols,rows=x.shape
    matrix=x.reshape(nt,-1).T
    u,s,vh=np.linalg.svd(matrix,full_matrices=False)
    shrunk=np.maximum(s-np.float32(beta),np.float32(0))
    out=(u@(shrunk[:,None]*vh)).T.reshape(x.shape).astype(np.complex64)
    return out,shrunk

class Oracle:
    def __init__(self,image,mask,dims,mu):
        rows,cols,nt=dims
        self.shape=(nt,cols,rows);self.mu=mu
        self.image=image.reshape(self.shape)
        self.mask=mask.reshape(self.shape)
        index=np.argmax(np.abs(image.real)+np.abs(image.imag))
        self.image=(self.image/np.abs(image[index].real)).astype(np.complex64)
        self.scale=np.float32(1/math.sqrt(rows*cols))
        self.b=self.forward(self.image)
        self.zero=self.backward(self.b)

    def forward(self,x):
        return (np.fft.fft2(x,axes=(1,2)).astype(np.complex64)*self.scale*self.mask).astype(np.complex64)

    def backward(self,x):
        count=self.shape[1]*self.shape[2]
        return (np.fft.ifft2(x*self.mask,axes=(1,2)).astype(np.complex64)*np.float32(count)*self.scale).astype(np.complex64)

    def run(self,iterations,alpha,beta,tau,sigma):
        scalar=np.float32
        alpha,beta,tau,sigma=map(scalar,(alpha,beta,tau,sigma))
        l=np.maximum(self.zero.real,scalar(0)).astype(np.complex64)
        s=np.zeros_like(l);w=np.zeros((3,)+l.shape,np.complex64)
        p=w.copy();q=np.zeros((6,)+l.shape,np.complex64);r=s.copy()
        lb=l.copy();sb=s.copy();wb=w.copy()
        axes=(1,2,0)
        first_input=first_output=None
        for i in range(1,iterations+1):
            a0=scalar(math.sqrt(2))*alpha
            alpha0=scalar(math.exp(float(scalar(i)/scalar(iterations)*scalar(np.log(a0*scalar(.5)))+scalar(iterations-i)/scalar(iterations)*scalar(np.log(a0)))))
            alpha1=scalar(math.exp(float(scalar(i)/scalar(iterations)*scalar(np.log(alpha*scalar(.5)))+scalar(iterations-i)/scalar(iterations)*scalar(np.log(alpha)))))
            lo,so,wo=l.copy(),s.copy(),w.copy()
            r=(r+sigma*(self.forward(lb+sb)-self.b))/(scalar(1)+sigma)
            gs=np.stack([grad(sb,axis,self.mu) for axis in axes])
            p-=sigma*(gs+wb)
            norm=np.sqrt(np.sum(np.abs(p)**2,axis=0))
            p/=np.maximum(scalar(1),norm/alpha1)[None,...]
            d=lambda j,axis:adj(wb[j],axis,self.mu)
            terms=np.stack([d(0,1),d(1,2),d(2,0),(d(0,2)+d(1,1))/scalar(2),(d(2,1)+d(0,0))/scalar(2),(d(1,0)+d(2,2))/scalar(2)])
            q-=sigma*terms
            norm=np.sqrt(np.sum(np.abs(q[:3])**2,axis=0)+scalar(2)*np.sum(np.abs(q[3:])**2,axis=0))
            q/=np.maximum(scalar(1),norm/alpha0)[None,...]
            z=self.backward(r)
            divp=adj(p[0],1,self.mu)+adj(p[1],2,self.mu)+adj(p[2],0,self.mu)
            s-=tau*(z+divp)
            g=lambda j,axis:grad(q[j],axis,self.mu)
            divq=np.stack([g(0,1)+g(3,2)+g(4,0),g(3,1)+g(1,2)+g(5,0),g(4,1)+g(5,2)+g(2,0)])
            w-=tau*(divq-p)
            l-=tau*z
            if i==1:first_input=l.copy()
            l,_=proximal(l,beta)
            if i==1:first_output=l.copy()
            sb=scalar(2)*s-so;wb=scalar(2)*w-wo;lb=scalar(2)*l-lo
        self.first_input,self.first_output=first_input,first_output
        return np.concatenate([l.ravel(),s.ravel()])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('directory',type=pathlib.Path);ap.add_argument('image');ap.add_argument('mask')
    ap.add_argument('--iterations',type=int,required=True)
    for name,value in [('alpha',.004),('beta',.5),('tau',.25),('sigma',.25),('mu',1.)]:ap.add_argument('--'+name,type=float,default=value)
    ap.add_argument('--skip-full-cpu',action='store_true')
    a=ap.parse_args();dims,image=ra(a.image);md,mask=ra(a.mask);assert dims==md
    nd,native=ra(a.directory/'native.ra');od,iris=ra(a.directory/'iris.ra');assert nd==od==dims+(2,)
    compare('complete native HIP versus IRIS L/S',iris,native,atol=2e-5,rtol=2e-4,l2_limit=1e-5)
    oracle=Oracle(image,mask,dims,a.mu)
    for stage,reference in [('normalized',oracle.image),('sampled',oracle.b),('zerofill',oracle.zero)]:
        for model in ['native','iris']:
            data=np.fromfile(a.directory/f'{model}.{stage}.bin',np.complex64).reshape(reference.shape)
            compare(f'{model} independent {stage}',data,reference)
    for model in ['native','iris']:
        x=np.fromfile(a.directory/f'{model}.prox_input.bin',np.complex64).reshape(oracle.shape)
        result=np.fromfile(a.directory/f'{model}.prox_output.bin',np.complex64).reshape(oracle.shape)
        expected,singular=proximal(x,a.beta)
        compare(f'{model} independent whole complex SVD/shrink/DGMM/GEMM proximal',result,expected)
        actual=np.fromfile(a.directory/f'{model}.singular.bin',np.float32)
        compare(f'{model} independent shrunk singular values',actual,singular)
    if not a.skip_full_cpu:
        expected=oracle.run(a.iterations,a.alpha,a.beta,a.tau,a.sigma)
        compare('native complete original-source NumPy primal-dual oracle',native,expected)
        compare('IRIS complete original-source NumPy primal-dual oracle',iris,expected)
    else:print('Full CPU iteration loop explicitly skipped; complete native GPU comparison and independent first-stage/proximal oracles still required')
    text=(a.directory/'iris_run.log').read_text()
    match=re.search(r'IRIS actual vendor calls FFT_forward=(\d+) FFT_backward=(\d+) SVD=(\d+) Cdgmm_64=(\d+) Cgemm_64=(\d+)',text)
    assert match and tuple(map(int,match.groups()))==(a.iterations+1,a.iterations+2,a.iterations,a.iterations,a.iterations)
    for model,result in [('native',native),('iris',iris)]:
        logs=(a.directory/f'{"hip" if model=="native" else "iris"}_run.log').read_text()
        recon=result[:image.size]+result[image.size:]
        def ser(x):return -20*np.log10(np.linalg.norm(np.abs(x)-np.abs(oracle.image.ravel()))/np.linalg.norm(np.abs(oracle.image.ravel())))
        for label,expected in [('zerofill',ser(oracle.zero.ravel())),('recon',ser(recon))]:
            match=re.search(r'The SER of '+label+r':\s+([-+0-9.]+) dB',logs)
            assert match and abs(float(match[1])-expected)<.011,(model,label,expected)
    print(f'PASS all {iris.size} complex L/S outputs, independent FFT/SVD/source oracles, both SER values, actual full vendor-call coverage',flush=True)

if __name__=='__main__':main()
