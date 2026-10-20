#!/usr/bin/env python3
"""Independent mathematical t-SNE/FIt-SNE oracle, Float64/complex128 CPU only.
Preserves source axis-ignored normalization as an explicitly named observable.
Fourier convolution uses NumPy FFT rather than transcribing GPU direct DFT loops.
"""
import pathlib,json,hashlib,argparse,sys,time
import numpy as np
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1]
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=pathlib.Path,required=True);a.add_argument('--iterations',type=int,default=1);a.add_argument('--perplexity',type=float,default=50);z=a.parse_args();o=z.output;o.mkdir(parents=True,exist_ok=True);start=time.monotonic();N=60000;K=32;B=130;J=3;H=B*J;F=2*H
 rawdist=np.fromfile(R/'src/tsne-cuda/data/mnist_faissed/distances',dtype='<f4').reshape(N,K);ind=np.fromfile(R/'src/tsne-cuda/data/mnist_faissed/indices',dtype='<i8').reshape(N,K);initial=np.loadtxt(R/'src/tsne-cuda/data/points.txt',dtype=np.float32).reshape(2,N);assert np.all(ind>=0)and np.all(ind<N)and initial.size==2*N
 dist=rawdist.astype(np.float64)/np.max(rawdist);beta=np.ones(N);lo=np.full(N,-np.inf);hi=np.full(N,np.inf)
 for it in range(200):
  w=np.exp(-beta[:,None]*dist);w[:,0]=np.where(dist[:,0]==0,0,w[:,0]);s=w.sum(axis=1);entropy=-(np.where(w>0,w*np.log(np.maximum(w,np.finfo(float).tiny)),0)).sum(axis=1)/s+np.log(s);diff=entropy-np.log(z.perplexity);done=np.abs(diff)<1e-4;more=(diff>0)&~done;less=(diff<=0)&~done;lo[more]=beta[more];hi[less]=beta[less];beta[more]=np.where(np.isinf(hi[more]),beta[more]*2,(beta[more]+hi[more])/2);beta[less]=np.where(np.isinf(lo[less]),beta[less]/2,(beta[less]+lo[less])/2)
  if done.all():break
 # Actual BroadcastMatrixVector ignores axis and always divides by vector[column].
 nonsym=w/s[np.arange(K)][None,:];canonical=w/s[:,None]
 ids=np.arange(N)[:,None];codes=(ids*N+ind).ravel();order=np.argsort(codes,kind='stable');sc=codes[order];rev=(ind*N+ids).ravel();left=np.searchsorted(sc,rev,'left');right=np.searchsorted(sc,rev,'right');prefix=np.r_[0,np.cumsum(nonsym.ravel()[order])];recip=prefix[right]-prefix[left];sym=(nonsym.ravel()+recip).reshape(N,K)/(2*N)
 fields={'input_distances':rawdist,'input_indices':ind.astype(np.float32),'input_points':initial,'normalized_distances':dist,'pij_nonsymmetric':nonsym,'pij_symmetric':sym,'entropy':entropy,'beta':beta,'canonical_row_probability':canonical};x=initial.astype(float).copy();g=np.ones_like(x);old=np.zeros_like(x);spacing=np.array([1/6,1/2,5/6]);den=np.array([np.prod(spacing[j]-np.delete(spacing,j))for j in range(J)])
 for step in range(z.iterations):
  charges=np.c_[np.ones(N),x[0],x[1],x[0]**2+x[1]**2];mn=x.min();mx=x.max();bw=(mx-mn)/B;xb=np.clip(((x[0]-mn)/bw).astype(int),0,B-1);yb=np.clip(((x[1]-mn)/bw).astype(int),0,B-1);xp=(x[0]-(mn+xb*bw))/bw;yp=(x[1]-(mn+yb*bw))/bw
  lx=np.array([np.prod(xp[:,None]-np.delete(spacing,j),axis=1)/den[j]for j in range(J)]);ly=np.array([np.prod(yp[:,None]-np.delete(spacing,j),axis=1)/den[j]for j in range(J)]);coef=np.zeros((H,H,4));ix=[];iy=[]
  for i in range(J):
   for j in range(J):np.add.at(coef,(xb*J+i,yb*J+j),(lx[i]*ly[j])[:,None]*charges)
  # Exact source zero-embedded centered squared-Cauchy generating kernel.
  ker=np.zeros((F,F));axis=np.arange(H)*bw/J;v=(1+axis[:,None]**2+axis[None,:]**2)**-2
  for sx in [-1,1]:
   for sy in [-1,1]:ker[np.ix_(H+sx*np.arange(H),H+sy*np.arange(H))]=v
  fin=np.zeros((4,F,F));fin[:,:H,:H]=np.moveaxis(coef,-1,0);fk=np.fft.rfft2(ker);fw=np.fft.rfft2(fin,axes=(-2,-1));product=fw*fk[None,:,:];fout=np.fft.irfft2(product,s=(F,F),axes=(-2,-1));tilde=np.moveaxis(fout[:,H:,H:],0,-1);potential=np.zeros((N,4))
  for i in range(J):
   for j in range(J):potential+=(lx[i]*ly[j])[:,None]*tilde[xb*J+i,yb*J+j]
  normvec=(1+x[0]**2+x[1]**2)*potential[:,0]-2*(x[0]*potential[:,1]+x[1]*potential[:,2])+potential[:,3];norm=normvec.sum()-N;rep=np.array([x[0]*potential[:,0]-potential[:,1],x[1]*potential[:,0]-potential[:,2]]);dx=x[:, :,None]-x[:,ind];q=sym/(1+(dx**2).sum(axis=0));workspace=dx*q[None,:,:];attr=workspace.sum(axis=2);grad=12*attr-rep/norm;g=np.where(np.signbit(grad)!=np.signbit(old),g+.2,g*.8);g=np.maximum(g,.01);old=.5*old-200*g*grad;x+=old
  fields.update(points=x.copy(),gains=g.copy(),old_forces=old.copy(),charges=charges,point_box_idx=yb*B+xb,x_in_box=xp,y_in_box=yp,x_interpolated=lx,y_interpolated=ly,w_coefficients=coef,kernel_tilde=ker,fft_kernel=fk,fft_input=fin,fft_product=product,fft_output=fout,y_tilde=tilde,potentials=potential,normalization_vec=normvec,normalization=np.array([norm]),workspace=workspace,attractive_forces=np.zeros_like(x),repulsive_forces=np.zeros_like(x));print(json.dumps({'CPU_iteration':step+1,'max_coordinate':float(np.max(np.abs(x))),'normalization':float(norm)}),flush=True)
 rows={}
 for name,v in fields.items():
  v=np.asarray(v,dtype='<c16'if np.iscomplexobj(v)else'<f8');assert np.isfinite(v).all();f=o/(name+'.bin');v.tofile(f);rows[name]={'shape':v.shape,'dtype':str(v.dtype),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
 m={'case':'source_mnist60000_knn32_boxes130','iterations':z.iterations,'perplexity_target':z.perplexity,'perplexity_search_iterations':it+1,'entropy_target_achieved_rows':int(done.sum()),'maximum_supported_perplexity':31,'CPU_seconds':time.monotonic()-start,'NumPy':np.__version__,'oracle_method':'Float64 vectorized entropy, sparse reciprocal lookup and Lagrange gridding; complex128 mathematical FFT convolution distinct fromsource naiveDFT loops. Source axisignored normalization retained, canonical rowcontrol separate.','files':rows,'source_recipe_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'source_quality_witness':{'conditional_row_sum_min':float(nonsym.sum(axis=1).min()),'conditional_row_sum_max':float(nonsym.sum(axis=1).max()),'default_target_unattainable':bool(z.perplexity>31)},'GPU_execution':False};(o/'oracle_manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'done':True,'seconds':m['CPU_seconds'],'files':len(rows),'bytes':sum(x['bytes']for x in rows.values())}))
if __name__=='__main__':main()
