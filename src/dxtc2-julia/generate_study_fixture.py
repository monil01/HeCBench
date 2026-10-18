from pathlib import Path
import struct,json
p=Path(__file__).resolve().parent
idx=[0]*16;perms=[]
for a in range(15,-1,-1):
 for m in range(a,16):idx[m]=2
 for b in range(15 if a==0 else 16,a-1,-1):
  if b<16:idx[b]=1
  perms.append(sum(v<<(2*i) for i,v in enumerate(idx)))
assert len(perms)==151
perms += [0xAA555]*9
idx=[0]*16
for a in range(15,-1,-1):
 for m in range(a,16):idx[m]=2
 for b in range(15 if a==0 else 16,a-1,-1):
  for m in range(b,16):idx[m]=3
  for c in range(15 if b==0 else 16,b-1,-1):
   if c<16:idx[c]=1
   if 3 in idx:perms.append(sum(v<<(2*i) for i,v in enumerate(idx)))
assert len(perms)==975
perms += [0xAAFF55]*49
image=[]
for b in range(12):
 for j in range(16):
  i=(j*(1+2*(b%7))+b)%16
  rgb=[20+(b%4)*5+i*(7+b%3),15+(b%3)*9+i*(4+b%4),30+(b%2)*12+i*(5+b%5)]
  assert all(0<=v<=255 for v in rgb)
  image.append(rgb[0]|rgb[1]<<8|rgb[2]<<16|((255 if j%2 else 0)<<24))
words=[12,12,0,5,2]+perms+image
p.joinpath('fixture.bin').write_bytes(struct.pack('<%sI'%len(words),*words))
p.joinpath('fixture_description.json').write_text(json.dumps(dict(blocks=12,paths=[dict(blocks=12,offset=0),dict(blocks=5,offset=2)],input_prefix_count=len(words),fixture='12nondegenerateaffineRGBline4x4tiles withpermutedpixelorder; alphaalternates0/255butignoredsource; secondpathsubrangeoffset2 count5 retainsothercompressedwords guardsentinel',independent_quality_contract='DecodeallactiveBC1blocks independently; fullRGBcomponentMSE<=256againstactualoriginalpixels forselectedboundedlinearRGBfixtures. Thisisfixture-specificlossyqualityproperty, notglobaloptimizeroptimality/fullimagequalityclaim.',permutation_generation='151threecluster+9padding+815fourcluster+49padding=1024tableentries; actualnativetableandoriginalJuliahosttablecheckedexactbeforemeasurement'),indent=2))
