#!/usr/bin/env python3
"""Original exact text plus named source-mode controls, independent byte oracle."""
from pathlib import Path
import argparse,json,hashlib,struct
P=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=Path('/tmp/hecbench-stencil-ports-ss-fixtures'));z=a.parse_args();F=z.output.resolve();original=F/'StringSearch_Input.txt';meta=json.load(open(P/'asset_provenance.json'));assert original.is_file()and hashlib.sha256(original.read_bytes()).hexdigest()==meta['assets'][0]['sha256'];assert not(F/'cases.json').exists(),'Refuseexistingcampaignfixtures'
W=131072;pattern=b'clEnqueueNDRangeKernel';boundaryN=2*W+259;boundary=bytearray(b'x'*(boundaryN+len(pattern)-1));positions=[0,255,511,W-1,W+31,boundaryN-1]
for i,pos in enumerate(positions):boundary[pos:pos+len(pattern)]=pattern.swapcase()if i%2 else pattern
one_miss=bytearray(b'a'*(2*W));one_miss[W-1]=ord('b')
cases=[('opencl12_default',original,pattern,20000,None),('opencl12_naive_a',original,b'a',1,None),('dense_overlap17',None,b'a'*17,1,b'a'*(2*W+16)),('dense_naive_one_miss',None,b'a',1,bytes(one_miss)),('two_char_filter_reject',None,b'ab'+b'c'*15,1,(b'ab'+b'x'*15)*8193),('ten_char_final_reject',None,b'abcdefghij'+b'c'*7,1,(b'abcdefghij'+b'x'*7)*8193),('boundary_mixed_case',None,pattern,1,bytes(boundary))]
fold=bytes(c+32 if 65<=c<=90 else c for c in range(256));records=[]
for k,(name,path,patt,repeat,data)in enumerate(cases):
 if path is None:path=F/(name+'.txt');assert not path.exists();path.write_bytes(data)
 raw=path.read_bytes();text=raw.translate(fold);p=patt.translate(fold);gold=[i for i in range(len(raw)-len(patt)+1)if text[i:i+len(p)]==p];total=len(raw)-len(patt)+1;groups=(total+W-1)//W;counts=[0]*groups
 for i in gold:counts[i//W]+=1
 oracle=F/(name+'_gold.u32');oracle.write_bytes(struct.pack('<'+'I'*len(gold),*gold));records.append(dict(id=k,name=name,text_path=str(path),text_bytes=len(raw),text_sha256=hashlib.sha256(raw).hexdigest(),pattern_ascii=patt.decode('ascii'),pattern_bytes=len(patt),repeats=repeat,mode='naive'if len(patt)==1 else'load_balance',total_search_positions=total,workgroups=groups,search_positions_per_group=W,oracle_path=str(oracle),oracle_sha256=hashlib.sha256(oracle.read_bytes()).hexdigest(),complete_naive_oracle_matches=len(gold),complete_group_counts=counts,oracle_method='Everyvalidbyte-start tested withindependentASCIIcasefoldedfullsliceequality; overlapsincluded, notsourceBMH/stagedfilter.'))
(F/'cases.json').write_text(json.dumps(records,indent=2)+'\n');(F/'cases.tsv').write_text(''.join(r['name']+'\t'+str(r['repeats'])+'\t'+r['text_path']+'\t'+r['pattern_ascii']+'\t'+r['oracle_path']+'\n'for r in records));print(json.dumps([{k:r[k]for k in ['name','text_bytes','pattern_bytes','repeats','complete_naive_oracle_matches','complete_group_counts']}for r in records],indent=2))
