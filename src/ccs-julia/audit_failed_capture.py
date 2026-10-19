#!/usr/bin/env python3
"""CPU-only full parser for a saved source CCS failure, including variable print length."""
import argparse,array,json,struct
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument('capture',type=Path);args=ap.parse_args()
 raw=(Path(__file__).resolve().parent/'fixtures.bin').read_bytes();pos=4;cases=[]
 for _ in range(struct.unpack_from('<I',raw)[0]):
  n,D,m,rep,ng,ne,thr,ov=struct.unpack_from('<6I2f',raw,pos);pos+=32
  gene=struct.unpack_from('<%df'%ng,raw,pos);pos+=ng*4
  expected=struct.unpack_from('<%dd'%ne,raw,pos);pos+=ne*8;cases.append((n,D,m,gene,expected))
 values=array.array('d');values.frombytes(args.capture.read_bytes());off=0;inputwrong=0
 for n,D,m,g,e in cases:
  expected=list(g)+[0.]*(2*m*(D+n)+3*m)
  inputwrong+=sum(x!=y for x,y in zip(values[off:off+len(expected)],expected));off+=len(expected)
 result=[]
 for n,D,m,g,e in cases:
  fixed=3*m*(D+n)+6*m;textn=int(values[off+fixed]);guardpos=off+fixed+1+textn
  wrong=sum(x!=y for x,y in zip(values[off:off+fixed],e[:fixed]))
  guardwrong=sum(x!=y for x,y in zip(values[guardpos:guardpos+128],[-777.]*64+[165.]*64))
  result.append(dict(n=n,D=D,fixed_output_wrong=wrong,printed_bytes=textn,expected_printed_bytes=int(e[fixed]),guard_wrong=guardwrong))
  off=guardpos+128
 if off!=len(values):raise ValueError('Trailing or truncated capture')
 print(json.dumps(dict(input_wrong=inputwrong,cases=result,consumed_values=off),indent=2))
if __name__=='__main__':main()
