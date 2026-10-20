#!/usr/bin/env python3
"""Independent exact Boolean/NPN gate for the preserved pre-evaluation unit."""
import argparse,pathlib,struct,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--native',type=pathlib.Path,required=True);p.add_argument('--julia',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();raw=a.native.read_bytes();n,pi,k=struct.unpack_from('<iii',raw);off=12+4*k
inputs=[]
for _ in range(4):inputs.append(struct.unpack_from('<'+'i'*(n+1),raw,off));off+=4*(n+1)
a0,b0,levels,refs=inputs;lib=raw[off:off+1019472];off+=1019472;expected=raw[off:];got=a.julia.read_bytes();assert expected==got
pos=4*4*(n+1);cuts=got[pos:pos+28*8*(n+1)];pos+=28*8*(n+1);selected=got[pos:pos+28*(n+1)];pos+=28*(n+1);best=struct.unpack_from('<'+'i'*(n+1),got,pos)
mask=(1<<(1<<pi))-1;values=[mask]+[sum(((x>>j)&1)<<x for x in range(1<<pi)) for j in range(pi)]
for node in range(pi+1,n+1):
 def literal(q):return values[q//2]^(mask if q&1 else 0)
 values.append(literal(a0[node])&literal(b0[node]))
def cfields(c):
 sign,*leaves=struct.unpack_from('<5i',c);w=struct.unpack_from('<I',c,20)[0];return leaves,(w>>27)&15,w&65535,bool(c[24])
def checkcut(root,c):
 leaves,count,truth,used=cfields(c)
 if not used:return 0
 assert count<=4 and all(0<=x<=n for x in leaves[:count])
 for assign in range(1<<pi):
  address=sum(((values[leaves[j]]>>assign)&1)<<j for j in range(count));assert (truth>>address)&1==(values[root]>>assign)&1,(root,assign,leaves,count,truth)
 return 1<<pi
checks=sum(checkcut(root,cuts[(8*root+s)*28:(8*root+s+1)*28])for root in range(n+1)for s in range(8))
def libint(offset,i):return struct.unpack_from('<i',lib,offset+4*i)[0]
rows=[]
for root in range(n+1):
 c=selected[28*root:28*(root+1)];leaves,count,truth,used=cfields(c)
 if not used:continue
 checks+=checkcut(root,c);assert count in (3,4)
 if count==3:leaves[3]=0
 phase=libint(0,truth);perm=libint(262144,truth);cl=libint(524672,truth);size=libint(786816,cl);subgraphs=libint(787704,cl);which=best[root];assert 0<=which<subgraphs
 rt=libint(788592,cl*20+which);functions=[]
 for j in range(4):
  leaf=leaves[libint(524288,perm*4+j)];functions.append(values[leaf]^(mask if (phase>>j)&1 else 0))
 for j in range(size):
  left=libint(806352,cl*60+j);right=libint(859632,cl*60+j);ca=libint(912912,cl*60+j);cb=libint(966192,cl*60+j);assert left<len(functions)and right<len(functions)
  functions.append((functions[left]^(mask if ca else 0))&(functions[right]^(mask if cb else 0)))
 f=functions[rt];inversion=(f&1)^(values[root]&1);out=f^(mask if inversion else 0);assert out==values[root],('independent NPN reconstruction mismatch',root,cl,which)
 checks+=1<<pi;rows.append(dict(root=root,class_id=cl,selected_subgraph=which,npn_graph_nodes=size,selected_root=rt,full_boolean_function_preserved=True))
result=dict(scope='Rewrite pre-evaluation subsystem unit, not full resyn2',all_full_raw_bytes_equal=True,raw_compared_bytes=len(got),native_sha256=hashlib.sha256(raw).hexdigest(),julia_sha256=hashlib.sha256(got).hexdigest(),all_live_cut_and_selected_NPN_functions_independently_checked=True,checked_boolean_values=checks,selected_candidates=rows,native_parameter_initialization='ExplicitzeroedNPNinactivefields/selectedCut/bestOut initialbuffers; namedunit, not untouchedCLIheap-state certification.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
