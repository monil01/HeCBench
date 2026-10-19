#!/usr/bin/env python3
import pathlib,sys,json,hashlib
p=pathlib.Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=True)
alphabet='ARNDCQEGHILKMFPSTWYV'
cases=[('ACDEFGHIKLMNPQRSTVWY','ACDEFGHIKLMNPQRSTVWY'),('VVVVARNDCQEGHILKFFFF','ARNDCQEGHILK'),('AAAAAAACCCCCCCGGGGGGG','AAAAAAAGGGGGGG'),('ARNDCQEGHILKMFPSTWYV'*8,'ARNDCQEGHILKMFPSTWYV'*6+'ARNDCQEG')]
for k in range(61):
 l=17+(k*19)%112;a=''.join(alphabet[(j*7+j//9+k*3)%20]for j in range(l+16));b=a[5:5+l];
 if k%4==0:b=b[:l//2]+'W'+b[l//2+1:]
 cases.append((a,b))
for name,which in [('reference.fasta',0),('query.fasta',1)]:
 text=''.join(f'>named_pair_{i}\n{c[which]}\n'for i,c in enumerate(cases));(p/name).write_text(text)
(p/'fixture.json').write_text(json.dumps({'cases':cases,'max_reference':max(map(lambda c:len(c[0]),cases)),'max_query':max(map(lambda c:len(c[1]),cases)),'pairs':len(cases),'scope':'Positive named synthetic protein alignment pairs; external source DVCdatasetabsent. No FASTA linewrapping/zero-score/long-query-hostallocation certification.','sha256':{n:hashlib.sha256((p/n).read_bytes()).hexdigest()for n in ['reference.fasta','query.fasta']}},indent=2)+'\n')
