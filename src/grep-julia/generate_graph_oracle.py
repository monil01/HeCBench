#!/usr/bin/env python3
"""CPU source-semantic graph/capacity control; independent match truth remains Python re."""
from pathlib import Path
import array,json,re,hashlib
P=Path(__file__).resolve().parent
CONCAT=27;ALT=4;Q=2;STAR=3;PLUS=1;OPEN=5;CLOSE=6;ANY=21

def postfix(tokens):
 out=[];stack=[];nalt=natom=0
 for c in tokens:
  if c==0:break
  if c==OPEN:
   if natom>1:natom-=1;out.append(CONCAT)
   stack.append((nalt,natom));nalt=natom=0
  elif c==ALT:
   assert natom>0
   while natom>1:natom-=1;out.append(CONCAT)
   natom-=1;nalt+=1
  elif c==CLOSE:
   assert stack and natom>0
   while natom>1:natom-=1;out.append(CONCAT)
   natom-=1;out.extend([ALT]*nalt);nalt,natom=stack.pop();natom+=1
  elif c in (Q,STAR,PLUS):assert natom>0;out.append(c)
  else:
   if natom>1:natom-=1;out.append(CONCAT)
   out.append(c);natom+=1
 assert not stack
 while natom>1:natom-=1;out.append(CONCAT)
 out.extend([ALT]*nalt);return out+[0]
def graph(post):
 nodes={101:[256,0,0]};frags=[];count=0
 def new(c,a=0,b=0):
  nonlocal count
  count+=1;assert count<=100;nodes[count]=[c,a,b];return count
 def patch(outs,target):
  for i,slot in outs:nodes[i][slot]=target
 for c in post[:-1]:
  if c==CONCAT:
   s2,o2=frags.pop();s1,o1=frags.pop();patch(o1,s2);frags.append((s1,o2))
  elif c==ALT:
   s2,o2=frags.pop();s1,o1=frags.pop();n=new(257,s1,s2);frags.append((n,o1+o2))
  elif c==Q:
   s,o=frags.pop();n=new(257,s,0);frags.append((n,o+[(n,2)]))
  elif c in (STAR,PLUS):
   s,o=frags.pop();n=new(257,s,0);patch(o,n);frags.append((n if c==STAR else s,[(n,2)]))
  else:
   n=new(258 if c==ANY else c);frags.append((n,[(n,1)]))
 assert len(frags)==1;start,outs=frags.pop();patch(outs,101);return nodes,count,start

def evaluate(nodes,start,text,stats):
 def add(result,start):
  pending=[start];seen=0
  while pending:
   stats['max_epsilon_stack']=max(stats['max_epsilon_stack'],len(pending));s=pending.pop();seen+=1;assert seen<10000,'epsilon cycle'
   if not s:continue
   c,a,b=nodes[s]
   if c==257:pending.extend([a,b])
   else:result.append(s)
  stats['max_active_list']=max(stats['max_active_list'],len(result));assert len(result)<=100
 current=[];add(current,start)
 for char in text:
  nxt=[]
  for s in current:
   c,a,b=nodes[s]
   if c==char or c==258:add(nxt,a)
  current=nxt
 return any(nodes[s][0]==256 for s in current)
plan=json.load(open(P/'study_plan.json'));unique=set()
for k in range(3):unique.update((P/f'text{k}_u8.bin').read_bytes()[:-1].split(b'\0'))
alltrace=[];summary=[]
for j,pat in enumerate(plan['patterns']):
 post=postfix((P/f'regex{j}_u8.bin').read_bytes());nodes,count,start=graph(post);one=[-999]*404;one[:3]=[len(post)-1,count,start];one[3:3+len(post)]=post
 for i in range(1,count+1):one[103+(i-1)*3:103+i*3]=nodes[i]
 stats={'max_epsilon_stack':0,'max_active_list':0};checks=0
 for text in unique:
  assert evaluate(nodes,start,text,stats)==bool(re.fullmatch(pat.encode(),text)),(pat,text);checks+=1
 alltrace +=[-12345]*16+one*512+[-12345]*16
 summary.append({'pattern':pat,'postfix_length':len(post)-1,'state_count':count,'start':start,'unique_full_string_checks':checks,**stats,'source_list_deduplication_added':False,'epsilon_cycle_observed':False})
(P/'oracle_graph_i32.bin').write_bytes(array.array('i',alltrace).tobytes());report={'benchmark':'grep','source_semantic_graph_control_not_independent_regex_oracle':True,'independent_regex_oracle':'Pythonre.fullmatch primaryallrawmaskbytes anduniquestringclosurecontrols','patterns':summary,'actual_source_capacities':{'postfix':8000,'nodes':100,'parser_parens':100,'fragment_stack':1000,'active_lists':100,'epsilon_stack':100},'fullgraph_values':len(alltrace),'oracle_graph_sha256':hashlib.sha256((P/'oracle_graph_i32.bin').read_bytes()).hexdigest()};(P/'graph_capacity_proof.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
