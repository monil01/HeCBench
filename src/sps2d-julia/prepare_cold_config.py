from pathlib import Path
import json
P=Path(__file__).resolve().parent;R=P.parents[1];c=json.loads((P/'study_config.json').read_text());old=c['models']['cuda']['cwd'].split('/src/sps2d-julia')[0]
def relocate(x):
 if isinstance(x,str):return x.replace(old,str(R))
 if isinstance(x,list):return [relocate(a)for a in x]
 if isinstance(x,dict):return {k:relocate(v)for k,v in x.items()}
 return x
c=relocate(c);c.pop('reuse_result',None);c.pop('reuse_models',None);c['configuration_id']+='_fresh_all4_reproduction';out=P/'study_config_cold.json';out.write_text(json.dumps(c,indent=2)+'\n');print(out)
