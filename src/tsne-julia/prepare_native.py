#!/usr/bin/env python3
"""Host-only observation adapter; original GPU definitions remain unmodified."""
import pathlib,json,hashlib
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1];O=R/'src/tsne-cuda'
def main():
 s=(O/'fit_tsne.cu').read_text();changes=[]
 def sub(old,new):
  nonlocal s
  assert old in s;changes.append({'before':old,'after':new});s=s.replace(old,new,1)
 sub('    // Max-norm the distances', '    observe("input_indices",knn_indices_device);\n    // Max-norm the distances')
 sub('    tsne::utils::MaxNormalizeDeviceVector(', '    observe("input_distances",knn_distances_device);\n    tsne::utils::MaxNormalizeDeviceVector(')
 sub('    // Search Perplexity','    observe("normalized_distances",knn_distances_device);\n    // Search Perplexity')
 sub('    // Symmetrize the pij matrix','    observe("pij_nonsymmetric",pij_nonsymmetric_device);\n    // Symmetrize the pij matrix')
 sub('    // Initialize Low-Dim Points','    // Initialize Low-Dim Points')
 sub('    points_host.clear();','    observe("input_points",points_device);\n    points_host.clear();')
 names=['pij_indices_device','pij_symmetric_device','attractive_forces_device','repulsive_forces_device','gains_device','old_forces_device','ones_device','normalization_vec_device','pij_workspace_device','points_device','point_box_idx_device','x_in_box_device','y_in_box_device','y_tilde_values','x_interpolated_values_device','y_interpolated_values_device','potentialsQij_device','w_coefficients_device','chargesQij_device','box_lower_bounds_device','box_upper_bounds_device','kernel_tilde_device','fft_kernel_tilde_device','fft_scratchpad_device','fft_input','fft_w_coefficients','fft_output','y_tilde_spacings_device','denominator_device']
 sub('    // Return some final values','    GpuErrorCheck(cudaDeviceSynchronize());\n'+''.join('    observe("'+n+'",'+n+');\n'for n in names)+'    observe_scalar("normalization",normalization);\n    // Return some final values')
 out=P/'observed_fit.inc';out.write_text(s);m={'original_fit_source':str(O/'fit_tsne.cu'),'original_sha256':hashlib.sha256((O/'fit_tsne.cu').read_bytes()).hexdigest(),'observed_host_source_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'changes':changes,'original_GPU_kernel_definitions_modified':False,'observation_time_excluded':True,'source_parsing_arithmetic_and_grid_preserved':True};(P/'native_diagnostics/extraction_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
if __name__=='__main__':main()
