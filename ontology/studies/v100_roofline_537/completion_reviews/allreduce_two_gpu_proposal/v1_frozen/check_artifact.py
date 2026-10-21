"""Independent complete bitwise checker for both-rank input/output/tmp/guards."""
import array,struct,pathlib,sys,hashlib
SIZES=(1,32,33,256,4099,1048576)
COUNT=1+len(SIZES)+2*sum(SIZES)+sum(2*n+2*((n+1)//2)+768 for n in SIZES)
def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def check(path):
 p=pathlib.Path(path)
 if p.stat().st_size!=COUNT*4:raise ValueError('completeartifactsize')
 words=array.array('I');words.frombytes(p.read_bytes())
 if sys.byteorder!='little':words.byteswap()
 pos=0
 def one(value):
  nonlocal pos
  if words[pos]!=bits(value):raise ValueError('bitwisemismatchword'+str(pos))
  pos+=1
 def pattern(n,values,start=0):
  nonlocal pos
  for i in range(n):
   if words[pos+i]!=values[(i+start)%len(values)]:raise ValueError('fullbitwisemismatchword'+str(pos+i))
  pos+=n
 rankpatterns=[[bits((r+1)*.5+(i-8)/32)for i in range(17)]for r in range(2)];summed=[bits(1.5+(i-8)/16)for i in range(17)];sentinel=[bits(-777)]
 one(6)
 for n in SIZES:
  one(n)
  for r in range(2):pattern(n,rankpatterns[r])
 for n in SIZES:
  for r in range(2):pattern(n,summed)
  half=(n+1)//2;tail=n-half
  for r in range(2):
   length=tail if r==0 else half;start=half if r==0 else 0;pattern(length,rankpatterns[1-r],start);pattern(half-length,sentinel)
  pattern(768,sentinel)
 if pos!=COUNT:raise ValueError('fullcount')
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return {'path':str(p),'bytes':p.stat().st_size,'count':COUNT,'checked_values':COUNT,'bitwise_full_both_ranks':True,'sha256':h.hexdigest()}
