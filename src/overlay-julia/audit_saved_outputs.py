"""CPU-only all-trial input/RGB/coverage audit; no GPU retiming."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
p=Path(__file__).resolve().parent;image=np.fromfile(p/'study_input_f32.bin',dtype='<f4');boxes=np.fromfile(p/'study_boxes_i32.bin',dtype='<i4');expected=np.fromfile(p/'study_expected_f64.bin',dtype='<f8');mask=np.fromfile(p/'study_coverage_u8.bin',dtype=np.uint8).astype(bool);prefix=np.concatenate([image.astype('<f8'),boxes.astype('<f8'),np.array([255,204,203,1,640,480,245760],dtype='<f8')]);assert len(prefix)==1904647 and len(expected)==921600 and np.isfinite(prefix).all();rawprefix=prefix.tobytes();rp=Path(sys.argv[1]);r=json.loads(rp.read_text());models={};source_image=image.reshape(-1,3)
for model,m in r['models'].items():
 trials=[]
 for trial in m['trials']:
  f=Path(trial['output_artifact']['saved_path']);raw=f.read_bytes();assert raw[:len(rawprefix)]==rawprefix;got=np.frombuffer(raw,dtype='<f8')[len(prefix):];diff=np.abs(got-expected);assert np.isfinite(got).all()and float(diff.max())<=1e-3;rgb=got.reshape(-1,3);assert np.array_equal(rgb[~mask],source_image[~mask].astype(np.float64));assert (rgb>=-1e-3).all()and(rgb<=255+1e-3).all()
  trials.append({'trial':trial['trial'],'artifact':str(f),'sha256':hashlib.sha256(raw).hexdigest(),'exact_input_prefix':True,'all_rgb_finite':True,'uncovered_pixels_unchanged':True,'max_abs_error':float(diff.max()),'max_rel_error_floor_0_001':float((diff/np.maximum(np.abs(expected),1e-3)).max())})
 models[model]=trials
out={'benchmark':'overlay','result':str(rp),'method':'Independent CPU rectangle-union Float64 single-blend oracle, all921600RGB/all20artifacts, actualimage/Int32box/color/shape prefix byteexact, uncovered pixels identical, finite/range checks. AllGPUsource245760launches retained.','models':models,'numpy':np.__version__};(p/'saved_artifact_audit.json').write_text(json.dumps(out,indent=2)+'\n');print({m:max(t['max_abs_error']for t in rows)for m,rows in models.items()})
