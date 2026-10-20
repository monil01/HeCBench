#!/usr/bin/env python3
"""Generate launch/observer adapters around byte-identical originalGPUbodies."""
import pathlib,re,json,hashlib
B=pathlib.Path(__file__).resolve().parent;H=B/'reference_history';C=B/'configured'
s=(H/'kernels.cu').read_text();original=s
names=re.findall(r'__global__ void (\w+)\(',s);assert len(names)==7

def end_brace(text,start,left='(',right=')'):
 depth=0
 for i in range(start,len(text)):
  if text[i]==left:depth+=1
  elif text[i]==right:
   depth-=1
   if depth==0:return i
 raise ValueError('Unbalanced')

def body(text,name):
 st=text.index('__global__ void '+name+'(');b=text.index('{',st);return text[b:end_brace(text,b,'{','}')+1]
sourcebindings=[];cases=[]
ptrs={'vals':0,'fbrLikeSlcInds':1,'dInds2':2,'dInds3':2,'fbrPtr0':3,'fbrPtr1':4,'fbrIdx1':5,'fbrPtr2':6,'fbrIdx2':7,'dU0':8,'dU1':9,'dU2':10,'dU3':11};scalars={'nFibers':'nfibers','mode':'mode','R':'rank','warpPerSlice':'wps','logOfWPC':'logWPC','fbrPerWarp':'fpw','logOfFPW':'logFPW'}
for op,name in enumerate(names):
 st=s.index(name+'<<<');lp=s.index('(',st);end=end_brace(s,lp);args=s[lp+1:end];launch=s[st:lp];assert '<<<grid, block, 0, 0>>>'in launch
 make='mm_make4'if name.endswith('_4D')else'mm_make3';replacement=f'mm_dispatch({make}({op},grid.x,block.x,{args}))';s=s[:st]+replacement+s[end+1:]
 sig=original[original.index('(',original.index('__global__ void '+name))+1:];sig=sig[:end_brace('('+sig,0)-1];params=[re.findall(r'\w+',p)[-1]for p in sig.split(',')];out=[]
 for p in params:
  if p in ptrs:
   cast='float*'if p=='dU0'else'const float*'if p=='vals'or p.startswith('dU')else'const size_t*';out.append(f'({cast})p.pointers[{ptrs[p]}]')
  else:out.append('p.'+scalars[p])
 cases.append(f' case {op}: {name}<<<p.grid,p.block,0,0>>>('+','.join(out)+');break;')
 sourcebindings.append({'kernel':name,'op':op,'original_body_sha256':hashlib.sha256(body(original,name).encode()).hexdigest(),'configured_body_sha256':hashlib.sha256(body(s,name).encode()).hexdigest(),'unchanged':body(original,name)==body(s,name)})
assert all(x['unchanged']for x in sourcebindings)
pos=s.index('int MTTKRP_MIHCSR_GPU(');s=s[:pos]+'void mm_native_launch(const MMLaunch&p){switch(p.op){\n'+'\n'.join(cases)+'\n default:throw std::runtime_error("Invalidnativeop");}}\n\n'+s[pos:];s='#include "../study_interface.h"\n'+s
s=s.replace('cudaMalloc(', 'mm_malloc(').replace('cudaFree(', 'mm_free(')
needle='    for (int m = 0; m < TiledX[0].ndims; ++m){';pos=s.index(needle,s.index('for (int MTTKRPmode'))
s=s[:pos]+'    mm_mode_begin(MTTKRPmode,dU,szDU,TiledX);\n'+s[pos:]
needle='''      cudaDeviceSynchronize();
    }
  }

  /* Copying output''';assert s.count(needle)==1;s=s.replace(needle,'''      cudaDeviceSynchronize();
    }
    mm_mode_end(MTTKRPmode,dU,szDU,TiledX);
  }

  /* Copying output''');(C/'kernels.cu').write_text(s)
s=(H/'main.cu').read_text();needle='  randomize_mats(X, U, Opt);';assert s.count(needle)==2;pos=s.index(needle)+len(needle);s=s[:pos]+'\n  mm_begin_case(X,U,Opt);'+s[pos:];s='#include "../study_interface.h"\n'+s;(C/'main.cu').write_text(s)
(B/'native_kernel_body_identity.json').write_text(json.dumps({'benchmark':'mmcsf','all7_original_GPU_bodies_unchanged':True,'rows':sourcebindings,'observer_hooks':['Mainafterinitialrandomizebeforeformatconstruction','GPUallocationregistering/exactviewlengths','Fullpremode/postmodefactorstate/canonicalreadonlyinputs'],'no_native_kernel_fallback_when_Julia_callback_set':True},indent=2)+'\n')
print('Prepared7launchinterceptors; originalGPUbodiesunchanged')
