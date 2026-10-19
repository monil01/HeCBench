"""Independent histogram probability audit; does not repeat the alias-table constructor."""
from pathlib import Path
import argparse,json,struct,math,hashlib
p=argparse.ArgumentParser();p.add_argument('--capture-dir',type=Path,required=True);p.add_argument('--spectrum',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();lines=a.spectrum.read_text().splitlines();nbin,lower,mode=lines[1].split();n=int(nbin);assert int(mode)==0 and n==145
pairs=[list(map(float,s.split()))for s in lines[2:]if s.strip()];assert len(pairs)==n
edges=[float(lower)]+[s[0]for s in pairs];mass=[s[1]for s in pairs];assert all(edges[k]<edges[k+1]for k in range(n)) and all(v>=0 for v in mass)
D=a.capture_dir;read=lambda name,fmt:list(struct.unpack('<'+fmt*(n+1),(D/name).read_bytes()));x=read('spectrum_xi.bin','f');w=read('spectrum_wi.bin','f');b=read('spectrum_alias.bin','i');f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
assert x==list(map(f32,edges));assert w[n]==b[n]==0;assert all(0<=v<=1 for v in w[:n]);assert all(0<=v<n for v in b[:n]);prob=[0.]*n
for j in range(n):prob[j]+=w[j]/n;prob[b[j]]+=(1-w[j])/n
expected=[v/math.fsum(mass)for v in mass];errors=[abs(v-t)for v,t in zip(prob,expected)];assert max(errors)<2e-8;assert abs(math.fsum(prob)-1)<1e-14
mean_expected=math.fsum(p*(edges[k]+edges[k+1])/2 for k,p in enumerate(expected));mean_actual=math.fsum(p*(x[k]+x[k+1])/2 for k,p in enumerate(prob))
report={'all_pass':True,'histogram_intervals':n,'edges':n+1,'edges_match_input_Float32_bits':True,'all_alias_targets_in_range':True,'unused_tail_zero':True,'max_probability_absolute_error':max(errors),'mean_energy_expected_MeV':mean_expected,'mean_energy_alias_MeV':mean_actual,'scope':'Analytic alias-marginal probabilities and actualdevice-prepared inputreadback; not MonteCarlo statisticaltest/fulltransport oracle. Corrected originalhostinit mode0only.','hashes':{q.name:hashlib.sha256(q.read_bytes()).hexdigest()for q in [a.spectrum,D/'spectrum_xi.bin',D/'spectrum_wi.bin',D/'spectrum_alias.bin']}};a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
