from pathlib import Path
import struct,json,hashlib,collections
B=Path(__file__).resolve().parent
MASK=(1<<64)-1
C1=0x87c37b91114253d5;C2=0x4cf5ad432745937f
def rotate(v,n):return ((v<<n)|(v>>(64-n)))&MASK
def mix(v):
 v^=v>>33;v=v*0xff51afd7ed558ccd&MASK;v^=v>>33;v=v*0xc4ceb9fe1a85ec53&MASK;return v^(v>>33)
def oracle(data,seed):
 a=b=seed
 full=len(data)//16*16
 for p in range(0,full,16):
  x=int.from_bytes(data[p:p+8],'little');y=int.from_bytes(data[p+8:p+16],'little');x=rotate(x*C1&MASK,31)*C2&MASK;a^=x;a=(rotate(a,27)+b)&MASK;a=(a*5+0x52dce729)&MASK;y=rotate(y*C2&MASK,33)*C1&MASK;b^=y;b=(rotate(b,31)+a)&MASK;b=(b*5+0x38495ab5)&MASK
 tail=data[full:];x=int.from_bytes(tail[:8],'little');y=int.from_bytes(tail[8:],'little')
 if len(tail)>8:b^=rotate(y*C2&MASK,33)*C1&MASK
 if tail:a^=rotate(x*C1&MASK,31)*C2&MASK
 a^=len(data);b^=len(data);a=(a+b)&MASK;b=(b+a)&MASK;a=mix(a);b=mix(b);a=(a+b)&MASK;b=(b+a)&MASK;return a,b
assert oracle(b'',0)==(0,0)
allbytes=bytearray();alloffsets=[];alllengths=[];gold=[];cases=[]
special=list(range(48))+[63,64,65,255,256,257,4095,4096,4097,9999]
for n in [257,4099]:
 for fixture in [0,1]:
  keydata=bytearray();offsets=[];lengths=[];out=[]
  for i in range(n):
   length=special[i] if i<len(special) else (i*61+fixture*17)%257
   if fixture:keydata.extend(bytes([0xa5])*(i*3%7))
   offsets.append(len(keydata));lengths.append(length)
   payload=bytes((j%256 if fixture==0 else (i*29+j*73+(j>>3)*19+0x87)&255) for j in range(length));keydata.extend(payload);out.extend(oracle(payload,i))
  offsets.append(len(keydata));cases.append(dict(n=n,fixture=fixture,byte_offset=len(allbytes),byte_count=len(keydata),offsets_offset=len(alloffsets),lengths_offset=len(alllengths),oracle_offset=len(gold),tail_histogram=dict(sorted(collections.Counter(x%16 for x in lengths).items())),offset_mod8_histogram=dict(sorted(collections.Counter(x%8 for x in offsets[:-1]).items())),length_min=min(lengths),length_max=max(lengths),total_key_bytes=sum(lengths),all_empty_seed0_oracle_zero=out[:2]==[0,0]));allbytes.extend(keydata);alloffsets.extend(offsets);alllengths.extend(lengths);gold.extend(out)
(B/'keys_u8.bin').write_bytes(allbytes)
for name,values,fmt in [('offsets_u32.bin',alloffsets,'I'),('lengths_u32.bin',alllengths,'I'),('oracle_u64.bin',gold,'Q')]: (B/name).write_bytes(struct.pack('<'+fmt*len(values),*values))
prefix=sum(c['byte_count']+32+(c['n']+33)+(c['n']+32)+(2*c['n']+32) for c in cases)
(B/'fixture_description.json').write_text(json.dumps(dict(cases=cases,guard_bytes=16,guard_words=16,input_prefix_count=prefix,complete_capture_count=prefix+sum(2*c['n']+32 for c in cases),internal_repeats=10,seed_rule='Sourceuint32keyindex0..n-1, notarbitraryseedbuffer. Selectedhighestseed4098, no high32bitseed/all2^32certification.',oracle='Python arbitraryprecision modular2^64 with int.from_bytes block/tail slices, independently structured from C++/Julia byteassembly/switchladder; full128bits, emptyseed0closedform.',fixtures='Packed sourcec%256bytes andgapped variedallbytevalues, sourcelegal offset/lengthtable; all16tailremainders, unalignedkeybases, zero length and9999Bmaximum. Gaps legalforkernel butnotoriginalhostcumulativepacking, explicitnamedcontrol.',timing='Sum4independent originalkernelhostlaunch/contextcompletion windows /R10. Setup/reset/copies/JIT/readback/CPUoracle outside, repeatedoverwrite. Fivecheckedtrials mean2-5; no sourceoriginalCLI RNGidentity/fullmillionkey claim.',scope='Fullfinite noncryptographic128-bit hashoperator, not cryptographicsecurity/collision/avalanche-quality or measuredmemorytraffic certification.'),indent=2)+'\n')
