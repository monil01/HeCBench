from pathlib import Path
import struct,json,hashlib
B=Path(__file__).resolve().parent
CAPACITY=4096
FIELDS=16
GUARD=0xA55A3CC3
def source(i,f,fixture):
 if fixture==0:return (i*0x9e3779b9+f*0x45d9f3b+0x12345)&0xffffffff
 return ((1<<((i+7*f)%32)) ^ ((i*0x10204081+f*0xabcdef01)&0xffffffff))&0xffffffff
def initial(i,f,fixture):return ((0xfedcba98 if fixture else 0x76543210) ^ (i*0x12345+f*0x13579bdf))&0xffffffff
def expected(i,f,fixture,n):return (initial(i,f,fixture)+(4096*source(i,f,fixture) if i<n else 0))&0xffffffff
if __name__=='__main__':
 for name,fn in [('source',source),('initial',initial)]:
  values=[fn(i,f,fixture) for fixture in range(2) for i in range(CAPACITY) for f in range(FIELDS)];data=struct.pack('<'+'I'*len(values),*values);(B/('fixture_'+name+'_aos_u32.bin')).write_bytes(data)
 (B/'fixture_description.json').write_text(json.dumps({'capacity':4096,'fields':16,'count':4096,'active_counts':[4096,4093],'fixtures':2,'layouts':['AoS16fields','SoA16planes_capacity4096'],'paths':8,'source_dtype':'UInt32','oracle':'Independent arbitrary-precision initial+4096*source modulo2^32; inactive tail unchanged; no duplicate iterativeGPUformula','capture':'All8paths actual source+actualinitial destination prefix, then all8fulloutputs;16UInt32guards eachside eachbuffer. Physical storage perpath retained.','scope':'Sourcefull4096capacity with originalCOUNT4096; additionallegal4093active-tail, notsourceCLIunusedtailallocation. Layout andfieldprecisionfaithful.','timing':'Sum8separately timed kernel-only hostwindows with contextcompletion; allreset/transfers/setup/readback outside eachwindow; perpath timings retained; R1 perpath, fivecheckedtrials andmean2-5'},indent=2)+'\n')
