"""Independent byte-packet encoder and decoder for GFC's exact uint64 bit stream."""
from pathlib import Path
import struct,json,hashlib,sys
P=Path(__file__).resolve().parent;N=512;NW=8;BLOCKS=4;WPB=2;CAP=N//2*17;CUTS=[64*(i+1)for i in range(NW)]
def encode(values,dim):
 out=bytearray([165])*CAP;offsets=[];start=0
 for stop in CUTS:
  pos=start//2*17;previous=[0]*32
  for base in range(start,stop,32):
   packet=bytearray(16);payload=bytearray()
   for lane in range(32):
    current=values[base+lane];old=previous[lane];negative=current<old;delta=abs(current-old);count=(delta.bit_length()+7)//8
    if count==2:count=3
    code=(8 if negative else 0)+(count-1 if count>=3 else count)
    packet[lane//2]|=code<<(4*(lane%2));payload.extend(delta.to_bytes(count,'little'))
   packet.extend(payload);out[pos:pos+len(packet)]=packet;pos+=len(packet);previous=[values[base+32-dim+lane%dim]for lane in range(32)]
  offsets.append(pos);start=stop
 return bytes(out),offsets
def decode(raw,offsets,dim):
 result=[];start=0
 for stop,end in zip(CUTS,offsets):
  pos=start//2*17;previous=[0]*32
  for base in range(start,stop,32):
   if pos+16>end:raise ValueError('Header out of chunk')
   header=raw[pos:pos+16];pos+=16;group=[]
   for lane in range(32):
    code=(header[lane//2]>>(4*(lane%2)))&15;count=code&7;count=count+1 if count>=2 else count
    if pos+count>end:raise ValueError('Payload out of chunk')
    delta=int.from_bytes(raw[pos:pos+count],'little');pos+=count;v=previous[lane]-delta if code&8 else previous[lane]+delta
    if not 0<=v<2**64:raise ValueError('Unsigned overflow')
    group.append(v)
   result.extend(group);previous=[group[32-dim+lane%dim]for lane in range(32)]
  if pos!=end:raise ValueError('Chunk offset disagreement')
  start=stop
 return result
def main():
 files={};cases=[]
 def save(name,data):
  f=P/name;f.write_bytes(data);files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 patterns=[0,1,255,256,65535,65536,0xffffff,0x1000000,0xffffffff,0x100000000,0xffffffffff,0x10000000000,0xffffffffffff,0x1000000000000,0xffffffffffffff,0x100000000000000,0xffffffffffffffff,0x8000000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000042,0xfff800000000ab12,0x3ff0000000000000,0xbff0000000000000,0x4000000000000000,0xc008000000000000,0x7fefffffffffffff,0x0010000000000000,0x000fffffffffffff,0x123456789abcdef,0xfedcba9876543210,0x3ff0000000000001]
 for case in(1,2):
  values=[struct.unpack('<Q',struct.pack('<d',float(i)))[0]for i in range(N)]if case==1 else[patterns[(i*13+i//32*7)%32]for i in range(N)]
  save(f'study_case{case}_u64.bin',struct.pack('<'+'Q'*N,*values))
  for dim in(1,2,3,32):
   raw,offs=encode(values,dim);assert decode(raw,offs,dim)==values
   packed=bytearray([BLOCKS,WPB,dim])+struct.pack('<i',N)+b''.join(struct.pack('<i',off-(start//2*17))for off,start in zip(offs,[0]+CUTS[:-1]))
   for start,off in zip([0]+CUTS[:-1],offs):packed.extend(raw[start//2*17:off])
   save(f'oracle_case{case}_d{dim}_payload.bin',raw);save(f'oracle_case{case}_d{dim}_off_i32.bin',struct.pack('<'+'i'*NW,*offs))
   cases.append(dict(case=case,dim=dim,output_capacity=CAP,absolute_offsets=offs,compressed_payload_bytes=sum(off-start//2*17 for off,start in zip(offs,[0]+CUTS[:-1])),source_file_bytes=len(packed),source_file_sha256=hashlib.sha256(packed).hexdigest(),CPU_roundtrip_exact=True))
 save('study_cuts_i32.bin',struct.pack('<'+'i'*NW,*CUTS))
 (P/'fixture_provenance.json').write_text(json.dumps(dict(cases=cases,files=files,n=N,num_warps=NW,blocks=BLOCKS,warps_per_block=WPB,cuts=CUTS,dimensions=[1,2,3,32],input_semantics='Source sequential binary64 values plus adversarial opaqueIEEE64bitpatterns including signedzeros/subnormals/infinity/NaNpayloads; codec performs integer operations, no floatingvalue-finiteness interpretation.',oracle='Independent sequential canonical packet encoder + independent decoder bitroundtrip; dimension predictor fromprior32-valuegroup, nibble sign/bytecount incl2→3 byte remapping.',generator=dict(path='src/fpdc-julia/generate_oracle.py',sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),python=sys.version),indent=2)+'\n')
 print(json.dumps(cases))
if __name__=='__main__':main()
