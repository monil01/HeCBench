#!/usr/bin/env python3
"""Plot frozen all537 coverage and strict family intensities (CPU, no installs).
Uses existing NumPy/Matplotlib; writes only a new /tmp directory. Input data,
plotted CSVs, immutable plotting source and all hashes accompany PDF/SVG.
"""
import argparse,collections,csv,hashlib,json,os,pathlib,sys
sys.dont_write_bytecode=True
P=pathlib.Path
MODELS=('cuda','cuda_jl','ka','jacc');LABELS={'cuda':'CUDA','cuda_jl':'CUDA.jl','ka':'KA','jacc':'JACC'}
CATS=('fp_position','no_counted_fp','source_na','hardware','pending_failed','other_unknown')
CATLABEL={'fp_position':'FP position observed','no_counted_fp':'No counted FP','source_na':'Source N/A','hardware':'Hardware blocked','pending_failed':'Pending / failed','other_unknown':'Other / unresolved'}
COLORS={'fp_position':'#277E79','no_counted_fp':'#DCC47A','source_na':'#9686B0','hardware':'#485361','pending_failed':'#DCE1E7','other_unknown':'#CA7964'}
FAMILY={'machine_learning':'Machine learning','signal_image':'Signal & image','domain_neutral_primitives':'General primitives','linear_algebra':'Linear algebra','hardware_data_movement':'Hardware & data movement','pde_continuum':'PDE / continuum','stochastic_statistics':'Statistics & sampling','graph_irregular':'Graphs & irregular','particle_molecular':'Particles & molecules','discrete_crypto':'Discrete / crypto','geometry_robotics':'Geometry & robotics','quantum_electronic':'Quantum / electronic','fluid_dynamics':'Fluid dynamics','molecular_simulation':'Molecular simulation'}
def sha(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def csvfile(p,rows,fields):
 with p.open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def category(r):
 c=r.get('assessment_category','');s=r.get('profiling_status','')
 if c=='hardware_prerequisite_blocked'or s=='native_hardware_blocked':return 'hardware'
 if c=='source_reviewed_runtime_without_profiled_device_kernels'or s=='roofline_not_applicable_no_device_kernels':return 'source_na'
 if c=='valid_selected_observations_no_counted_FP':return 'no_counted_fp'
 q=r.get('qualified_precision_tier_projections')
 if c=='valid_selected_observations'and isinstance(q,(int,float))and q>0:return 'fp_position'
 if c=='no_valid_selected_counter_capture'or s in ['pending_profile_collection','not_started_plan_blocked','not_started_replay_method_blocked','collection_failed','profile_process_failed','profile_completed_counter_derivation_failed']:return 'pending_failed'
 return 'other_unknown'
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-root',type=P,default=P(__file__).resolve().parents[1]);ap.add_argument('--descriptor',type=P);ap.add_argument('--family-audit',type=P);ap.add_argument('--output-root',type=P,required=True);ap.add_argument('--snapshot-label',default='21 Oct 2026; partial campaign');a=ap.parse_args();root=a.source_root.resolve();study=root/'ontology/studies/v100_roofline_537';descriptor=(a.descriptor or study/'benchmark_assessment_snapshot/descriptor_assessment.json').resolve();audit=(a.family_audit or study/'retained_roi_aggregates/audit.json').resolve();out=a.output_root.absolute()
 if not out.resolve().is_relative_to(P('/tmp'))or out.resolve()==P('/tmp')or out.exists()or out.is_symlink():ap.error('Fresh output directory strictly below /tmp required')
 inputs={str(p):{'sha256':sha(p),'bytes':p.stat().st_size}for p in [descriptor,audit]};d=json.loads(descriptor.read_text());f=json.loads(audit.read_text());records=d['records'];names=sorted({r['benchmark']for r in records})
 if len(names)!=537 or len(records)!=2148 or len({(r['benchmark'],r['model'])for r in records})!=2148:raise ValueError('Descriptor must contain exactly537 distinct names x4models')
 if set(r['model']for r in records)!=set(MODELS):raise ValueError('Unexpected model universe')
 snapshotmanifest=descriptor.parent/'snapshot_manifest.json'
 if snapshotmanifest.exists():inputs[str(snapshotmanifest)]={'sha256':sha(snapshotmanifest),'bytes':snapshotmanifest.stat().st_size}
 classifications=[{'benchmark':r['benchmark'],'model':r['model'],'scientific_family':r.get('scientific_family')or'unknown','assessment_category':r.get('assessment_category'),'profiling_status':r.get('profiling_status'),'plot_category':category(r),'qualified_precision_tier_projections':r.get('qualified_precision_tier_projections'),'observed_kernel_count':r.get('valid_selected_kernel_observations')}for r in records]
 bymodel={m:collections.Counter(x['plot_category']for x in classifications if x['model']==m)for m in MODELS};family_names=collections.defaultdict(set)
 for r in classifications:family_names[r['scientific_family']].add(r['benchmark'])
 families=sorted(family_names,key=lambda x:(-len(family_names[x]),FAMILY.get(x,x)))
 statusrows=[{'model':m,'category':c,'names':bymodel[m][c],'denominator':537}for m in MODELS for c in CATS];coverage=[]
 for family in families:
  for m in MODELS:
   rs=[r for r in classifications if r['scientific_family']==family and r['model']==m];n=len(rs)
   if n!=len(family_names[family]):raise ValueError('Scientific family differs across model denominator')
   count=collections.Counter(r['plot_category']for r in rs);q=count['fp_position'];missing=count['hardware']+count['pending_failed']+count['other_unknown']
   coverage.append({'scientific_family':family,'model':m,'denominator_names':n,'qualified_names':q,'qualified_fraction':q/n if n else None,'unmeasured_or_unresolved_names':missing,'no_counted_fp_names':count['no_counted_fp'],'source_na_names':count['source_na'],'hardware_names':count['hardware'],'pending_failed_names':count['pending_failed'],'other_unknown_names':count['other_unknown']})
 strict={(r['benchmark'],r['model']):r for r in f['records']if r.get('accepted_complete_scientific_kernel_family')is True};intensity=[];missing_family=[]
 for b in ['triad','babelstream','jacobi']:
  for m in MODELS:
   r=strict.get((b,m))
   if not r:missing_family.append({'benchmark':b,'model':m,'reason':'No admitted strict family; never a zero point'});continue
   t=r['totals'];v=t.get('fp32_flop_per_DRAM_byte');count=t.get('actual_scientific_launches')
   if not isinstance(v,(int,float))or v<=0:raise ValueError('Admitted family lacks positive integrated intensity')
   if v!=t['fp32_executed_flops']/t['dram_bytes']:raise ValueError('Integrated intensity is not exact ratio of totals')
   intensity.append({'benchmark':b,'model':m,'configuration_id':r['configuration_id'],'actual_scientific_launches':count,'fp32_executed_flops':t['fp32_executed_flops'],'dram_bytes':t['dram_bytes'],'integrated_fp32_flop_per_DRAM_byte':v,'sum_kernel_replay_seconds':t['sum_device_kernel_replay_seconds'],'source_result_sha256':r['capture_provenance'].get('source_result_sha256',r.get('original_numerical_receipt',{}).get('result_sha256')),'counter_sha256':r['capture_provenance']['counter_sha256'],'complete_elapsed_ROI':False,'causal_bottleneck':'unknown'})
 for p,h in inputs.items():
  if sha(p)!=h['sha256']:raise ValueError('Input changed before plotting '+p)
 out.mkdir(parents=True);os.environ['MPLCONFIGDIR']=str(out/'matplotlib_cache');os.environ['MPLBACKEND']='Agg'
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import Patch
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':10,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':240})
 fig=plt.figure(figsize=(7.5,7.1));gs=fig.add_gridspec(2,1,height_ratios=[1.2,3.7],hspace=.40,left=.36,right=.98,top=.91,bottom=.085);ax=fig.add_subplot(gs[0]);y=np.arange(4)
 left=np.zeros(4)
 for c in CATS:
  values=np.array([bymodel[m][c]for m in MODELS]);ax.barh(y,values,left=left,height=.61,color=COLORS[c],edgecolor='white',linewidth=.5,label=CATLABEL[c])
  for i,n in enumerate(values):
   if n>=16:ax.text(left[i]+n/2,y[i],str(n),ha='center',va='center',fontsize=8,color='white'if c in ['fp_position','hardware']else'#23313A')
  left+=values
 ax.set_yticks(y);ax.set_yticklabels([LABELS[m]for m in MODELS]);ax.invert_yaxis();ax.set_xlim(0,537);ax.set_xticks([0,100,200,300,400,537]);ax.set_xlabel('Benchmark names (537 per model)',labelpad=3);ax.tick_params(axis='y',length=0);ax.set_title('a  Selected-profile assessment',loc='left',pad=8)
 for spine in ['left','bottom']:ax.spines[spine].set_visible(False)
 ax.set_axisbelow(True);ax.grid(axis='x',color='#E8EBEF',linewidth=.5)
 handles=[Patch(facecolor=COLORS[c],label=CATLABEL[c]+(' ('+str(bymodel[MODELS[0]][c])+'/model)'if c=='hardware'and len({bymodel[m][c]for m in MODELS})==1 and bymodel[MODELS[0]][c]>0 else''))for c in CATS];fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.065,.995),ncol=3,frameon=False,columnspacing=1.4,handlelength=1.4)
 bx=fig.add_subplot(gs[1]);lookup={(r['scientific_family'],r['model']):r for r in coverage};matrix=np.array([[lookup[x,m]['qualified_fraction']for m in MODELS]for x in families]);im=bx.imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto')
 bx.set_yticks(np.arange(len(families)));bx.set_yticklabels([FAMILY.get(x,x.replace('_',' ').capitalize())+'  (n='+str(len(family_names[x]))+')'for x in families]);bx.set_xticks(np.arange(4));bx.set_xticklabels([LABELS[m]for m in MODELS]);bx.tick_params(length=0);bx.set_title('b  Names with FP positions; unmeasured / unresolved shown below',loc='left',pad=9,fontsize=9)
 for i,family in enumerate(families):
  for j,m in enumerate(MODELS):
   r=lookup[family,m];ink='white'if r['qualified_fraction']>.63 else'#173247';bx.text(j,i-.14,str(r['qualified_names'])+'/'+str(r['denominator_names']),ha='center',va='center',fontsize=8,color=ink);bx.text(j,i+.23,str(r['unmeasured_or_unresolved_names'])+' unmeas.',ha='center',va='center',fontsize=6.7,color=ink)
 bx.set_xticks(np.arange(-.5,4,1),minor=True);bx.set_yticks(np.arange(-.5,len(families),1),minor=True);bx.grid(which='minor',color='white',linewidth=.6);bx.tick_params(which='minor',bottom=False,left=False)
 fig.text(.065,.034,'FP geometry may be partial or source-unassigned; no whole-ROI or causal classification.\nUnmeasured excludes known no-FP and source-N/A cases. '+a.snapshot_label,fontsize=7,color='#4B5660',va='bottom')
 for suffix in ['pdf','svg','png']:fig.savefig(out/('coverage_assessment.'+suffix),metadata={'Creator':'plot_v100_benchmark_assessment.py'})
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(6.3,3.05));fig.subplots_adjust(left=.13,right=.98,top=.75,bottom=.30);MC={'cuda':'#283746','cuda_jl':'#3575AA','ka':'#26928B','jacc':'#C47737'};MARK={'cuda':'o','cuda_jl':'s','ka':'^','jacc':'D'};offsets=np.linspace(-.18,.18,4)
 for j,m in enumerate(MODELS):
  xs=[];ys=[]
  for i,b in enumerate(['triad','babelstream','jacobi']):
   r=next((x for x in intensity if x['benchmark']==b and x['model']==m),None)
   if r:xs.append(i+offsets[j]);ys.append(r['integrated_fp32_flop_per_DRAM_byte'])
   else:ax.annotate('N/A',(i+offsets[j],.02),xycoords=('data','axes fraction'),rotation=90,fontsize=6,ha='center')
  ax.scatter(xs,ys,label=LABELS[m],color=MC[m],marker=MARK[m],s=36,zorder=3,edgecolor='white',linewidth=.4)
 ax.set_yscale('log');ax.set_ylabel('Executed FP32 FLOP / DRAM byte');ax.set_xticks([0,1,2]);ax.set_xticklabels(['Triad\n100 launches / model','STREAM dot\n10 launches / model','Jacobi stencil\n20 launches / model']);ax.set_xlim(-.48,2.48);ax.grid(axis='y',which='major',color='#E2E6EA',linewidth=.6);ax.set_axisbelow(True);ax.legend(loc='lower left',bbox_to_anchor=(-.02,1.03),ncol=4,frameon=False,columnspacing=1.5);fig.suptitle('Integrated intensity of source-counted scientific families',fontsize=10,x=.53,y=.98);fig.text(.13,.065,'Cold kernel replay: sum(FLOPs) / sum(DRAM bytes), not averaged kernel intensities.\n'+str(len(intensity))+' admitted captures; not full elapsed ROI, unprofiled throughput, or a causal bottleneck.',fontsize=7,color='#4B5660')
 for suffix in ['pdf','svg','png']:fig.savefig(out/('paired_family_intensity.'+suffix),metadata={'Creator':'plot_v100_benchmark_assessment.py'})
 plt.close(fig)
 csvfile(out/'plotted_status_counts.csv',statusrows,['model','category','names','denominator']);csvfile(out/'plotted_name_categories.csv',classifications,['benchmark','model','scientific_family','assessment_category','profiling_status','plot_category','qualified_precision_tier_projections','observed_kernel_count']);csvfile(out/'plotted_family_coverage.csv',coverage,list(coverage[0]));csvfile(out/'plotted_integrated_intensity.csv',intensity,list(intensity[0])if intensity else['benchmark','model','integrated_fp32_flop_per_DRAM_byte'])
 source=P(__file__).resolve();snapshot=out/'plot_source_snapshot.py';snapshot.write_bytes(source.read_bytes());inputcopies=[]
 for p,name in [(descriptor,'descriptor_input.json'),(audit,'strict_family_audit_input.json')]:
  raw=p.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=inputs[str(p)]['sha256']:raise ValueError('Input changed while plotting')
  (out/name).write_bytes(raw);inputcopies.append({'original':str(p),'copy':str(out/name),'sha256':inputs[str(p)]['sha256']})
 for p,h in inputs.items():
  if sha(p)!=h['sha256']:raise ValueError('Input changed during plotting '+p)
 provenance={'schema_version':1,'snapshot_label':a.snapshot_label,'input_bindings':inputs,'input_copies':inputcopies,'plotter_source':str(source),'plotter_source_sha256':sha(source),'immutable_source_copy':str(snapshot),'interpreter':sys.executable,'python_version':sys.version,'numpy_version':np.__version__,'matplotlib_version':matplotlib.__version__,'denominators':{'names':537,'models':4,'model_slots':2148},'status_counts':{m:dict(bymodel[m])for m in MODELS},'unknown_categories_preserved':[r for r in classifications if r['plot_category']=='other_unknown'],'strict_family_missingness':missing_family,'plotted_admitted_family_count':len(intensity),'cold_replay':True,'whole_elapsed_ROI_claim':False,'causal_bottleneck_claim':False,'speedup_or_correlation_claim':False,'no_equal_kernel_intensity_averaging':True,'GPU_build_tests_install_Git':False};dump(out/'provenance.json',provenance)
 (out/'README.md').write_text('''# Frozen V100 assessment figures\n\ncoverage_assessment: topbars keep537names/model, separating observed FPpositions,\nno countedFP, source-reviewedN/A, hardware, pending/failed and unknowncategories.\nHeatmap cells are qualifiednames / familydenominator and count of unmeasured or\nunresolvednames (hardware + pending/failed + unknown); knownnoFP/sourceN/A are\nnot calledmissing. A qualifiedposition can be a partial/warmup/unassigned kernel\nsample, never completeROI or causal proof. Color is fractionofbenchmarknames,\nnot fractionofkernels. All categories survive in exact plottedCSVs.\n\npaired_family_intensity: the12strict admitted captures only, sumexecutedFP32FLOPs\n/ sumactualcoldreplayDRAMbytes. No equalintensityaverage, wholehostROI,\nunprofiledperformancespeedup, correlation or causalclaim. Jacobi excludes timed\nreset/copy/hostnormwork; STREAM isdotonly, notallsixoperators.\n\nBothPDF/SVG figures are CPU-only, createdwith existingMatplotlib; exactplotted\nCSVs, completeinputcopies, sourceSnapshot and input/outputSHAmanifest retained.\nNo benchmark sources/measurements/sharedmetadata were changed. No installations,\nGPU, builds, tests or Git. Snapshot is dated partialcampaign, not currentfuture\ncompletion. Defaultinputs pointto frozenstudy descriptors/strictaudit; override\nwith --descriptor/--family-audit. --output-root mustbefreshstrictlyunder/tmp.\n''')
 outputs=[p for p in out.iterdir()if p.is_file()];dump(out/'manifest.json',{'files':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}for p in sorted(outputs)],'source_snapshot_sha256':sha(snapshot),'input_sha256':{p:v['sha256']for p,v in inputs.items()}});print(json.dumps({'output':str(out),'model_slots':2148,'strict_family_captures':len(intensity),'unknown_category_names':sum(x['plot_category']=='other_unknown'for x in classifications)}))
if __name__=='__main__':main()
