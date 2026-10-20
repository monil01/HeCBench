#!/usr/bin/env python3
"""Independent CPU-reference power-of-two units, never fitted to GPU samples."""
import pathlib,json,hashlib,argparse,math
import numpy as np
P=pathlib.Path(__file__).resolve().parent
def main():
 a=argparse.ArgumentParser();a.add_argument('--oracle',type=pathlib.Path,required=True);a.add_argument('--output',type=pathlib.Path,required=True);z=a.parse_args();m=json.load(open(z.oracle/'oracle_manifest.json'));F=780;H=390
 product=np.fromfile(z.oracle/'fft_product.bin',dtype='<c16').reshape(4,F,H+1)
 full=np.empty((4,F,F),dtype=np.complex128);full[:,:,:H+1]=product;full[:,:,H+1:]=np.conj(product[:,(-np.arange(F))%F][:,:,np.arange(H-1,0,-1)])
 # Source inverse-first-axis temporary is this inverse transform transposed.
 scratch=np.fft.ifft(full,axis=2)*F
 fk=np.fromfile(z.oracle/'fft_kernel.bin',dtype='<c16');fo=np.fromfile(z.oracle/'fft_output.bin',dtype='<f8')*F*F
 arrays={'fft_kernel_tilde_device':fk,'fft_w_coefficients':product,'fft_output':fo,'fft_scratchpad_device':scratch};rows={}
 for name,x in arrays.items():
  peak=max(float(np.max(abs(x.real))),float(np.max(abs(x.imag))))if np.iscomplexobj(x)else float(np.max(abs(x)))
  exponent=math.ceil(math.log2(peak))if peak>0 else 0;rows[name]={'divide_by_power_of_two_exponent':exponent,'CPU_reference_component_peak':peak,'unit_scale':2.**exponent,'mathematical_reference':'Complex128FFTconvolution; inversefirstaxis scratch follows fullHermitian spectrum then1Dinverse. GPU samples not consulted.'}
 out={'benchmark':'tsne','configuration_scope':'originalMNIST60000/K32/130boxes/onecompleteiteration','contract_status':'prospective_revised_normwise_Fourier_after_preserved_pointwise_rejection','fields':rows,'unlisted_fields':'unchangedrawFloat32, including actualinputprefix andintegerindices','source_precision':'EveryrawGPUFloat32bitmustreproduce exactly after inversecodec; rejectunderflow/overflow/roundtripdifference.','comparison_budget_normalized':{'atol':.0002,'rtol':.0002},'CPU_independent_original_budgets_unchanged':{'atol':1e-6,'field_peak_scale':.003},'source_oracle_manifest_sha256':hashlib.sha256((z.oracle/'oracle_manifest.json').read_bytes()).hexdigest(),'source_recipe_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'GPU_execution':False,'NumPy':np.__version__};z.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
