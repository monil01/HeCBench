#!/usr/bin/env python3
"""Join every retained pipeline field/guard to its primary serialized artifact."""
import argparse,array,hashlib,json,mmap,struct,re
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('result');parser.add_argument('--model',required=True);parser.add_argument('--raw-root',required=True);parser.add_argument('--output',required=True);parser.add_argument('--process-id',type=int);args=parser.parse_args()
r=json.load(open(args.result));records=[]
for trial in r['models'][args.model]['trials']:
 primary=array.array('d');primary.frombytes(Path(trial['output_artifact']['saved_path']).read_bytes());prefix=[];outputs=[];shapes=set();launches=0;guard_total=0;raw_total=0;cases=[]
 # CPU semantic sections are checked structurally/finite and compared fully by central verifier.
 output_at=6329529
 for frames in (1,3,100):
  prefix.append(float(frames));paths=[p for p in Path(args.raw_root).glob(f'{args.model}_*_trial{trial["trial"]}_case{frames}.layout.jsonl') if re.match(re.escape(args.model)+r'_\d+_',p.name) and (args.process_id is None or p.name.startswith(f'{args.model}_{args.process_id}_'))];assert len(paths)==1,paths;base=str(paths[0]).removesuffix('.layout.jsonl');layout=[json.loads(x) for x in paths[0].read_text().splitlines()];timings=[json.loads(x) for x in Path(base+'.timings.jsonl').read_text().splitlines()];assert len(timings)==frames+2;launches+=len(timings)
  guard=Path(base+'.guards.u8').read_bytes();assert set(guard)=={165};guard_total+=len(guard);rawfile=open(base+'.raw.bin','rb');raw=mmap.mmap(rawfile.fileno(),0,access=mmap.ACCESS_READ);raw_total+=len(raw);previous={};timing_by_frame={v['frame']:v for v in timings if v['stage']=='MGVF'}
  def output(values):
   global output_at
   for value in values:
    assert primary[output_at]==value,(trial['trial'],frames,output_at,primary[output_at],value);output_at+=1
  for entry in layout:
   kind=entry['kind']
   if kind=='video_frame':
    at=entry['raw_offset_bytes'];N=entry['payload_bytes'];data=raw[at:at+N];assert N==307200
    prefix.extend([entry['frame'],entry['width'],entry['height'],entry['cropped'],entry['scaled']]);prefix.extend(float(int.from_bytes(data[i:i+6],'little')) for i in range(0,N,6))
   elif kind=='gpu_buffers':
    fields={}
    for field in entry['buffers']:
     at=field['raw_offset_bytes'];N=field['payload_bytes'];payload=raw[at+256:at+256+N];values=array.array('i' if field['dtype']=='int32' else 'f');values.frombytes(payload);fields[field['name']]=values
     offset=field['guard_offset_bytes'];actual=raw[at:at+256]+raw[at+256+N:at+512+N];assert actual==guard[offset:offset+512]
    stage=entry['stage'];frame=entry['frame'];before=entry['phase']=='before'
    if before:
     previous[(stage,frame)]=fields
     if stage=='GICOV':
      prefix.extend([219,640,175,596,256,408]);assert fields['d_gicov']==array.array('f',[0])*140160
      for name in ('d_cos_angle','d_grad_x','d_grad_y','d_sin_angle','d_tX','d_tY'):prefix.extend(fields[name])
     elif stage=='DILATION':
      prefix.extend([25,25,219,640,176,797]);prefix.extend(fields['d_strel']);output([2,frame,219,640]);output(fields['d_gicov'])
     else:
      prefix.extend([1.,1.,struct.unpack('<f',struct.pack('<f',1e-10))[0],struct.unpack('<f',struct.pack('<f',1e-5))[0],500]);ms=fields['d_m_array'];ns=fields['d_n_array'];offs=fields['d_I_offsets'];cells=len(ms);assert cells==36;offset=0
      for m,n,off in zip(ms,ns,offs):assert off==offset;offset+=m*n;shapes.add((m,n))
      assert offset==len(fields['d_I_all']);assert fields['d_I_all']==fields['d_IMGVF_all'];output([3,frame,cells])
      for name in ('d_I_offsets','d_m_array','d_n_array','d_I_all'):output(fields[name])
    elif stage=='GICOV':output([1,frame]);output(fields['d_gicov'])
    elif stage=='DILATION':
     output([2,frame]);output(fields['d_img_dilated']);assert primary[output_at:output_at+2]==array.array('d',[4,36]);output_at+=2+72
    else:
     cells=len(fields['d_m_array']);assert fields['d_iterations'].tolist()==timing_by_frame[frame]['iterations'];assert all(1<=x<=500 for x in fields['d_iterations']);output([3,frame,cells]);output(fields['d_IMGVF_all']);assert primary[output_at:output_at+4]==array.array('d',[5,frame,cells,20]);output_at+=4+22*cells
  cases.append({'frames':frames,'raw_path':base+'.raw.bin','raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),'guard_path':base+'.guards.u8','actual_guard_bytes':len(guard),'guard_sha256':hashlib.sha256(guard).hexdigest(),'GPU_launches':len(timings),'cell_iteration_sum':sum(sum(x['iterations']) for x in timings)})
  raw.close();rawfile.close()
 assert output_at==len(primary),(output_at,len(primary));assert len(prefix)==6329529;assert array.array('d',prefix)==primary[:len(prefix)]
 records.append({'trial':trial['trial'],'actual_guard_bytes':guard_total,'raw_bytes':raw_total,'GPU_launches':launches,'complete_primary_values':len(primary),'exact_prefix_values':len(prefix),'distinct_patch_shapes':len(shapes),'cases':cases})
receipt={'pass':True,'model':args.model,'run_id':r['run_id'],'checked_trials':len(records),'actual_guard_bytes':sum(x['actual_guard_bytes'] for x in records),'raw_bytes':sum(x['raw_bytes'] for x in records),'records':records,'scope':'Exact joins of all actual GPU buffers/inputs/outputs/guard bytes and primary external video/constant prefix; complete CPU trajectory outputs structurally/finite and compared separately by verifier. Independent operator witness scopes remain separately documented.'}
Path(args.output).write_text(json.dumps(receipt,indent=2)+'\n');print({k:v for k,v in receipt.items() if k!='records'})
