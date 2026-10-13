#pragma once
#include "binding_types.h"
inline void bind_all(){
  submitKernel("egs_bind_core",dim3(1),dim3(1),{memory(d_MT_tables),memory(d_MT_params),memory(d_MT_statuses),memory(d_stack.a),memory(d_stack.b),memory(d_stack.c),memory(d_total_step_counts),memory(d_total_list_depth),memory(d_total_num_inner_iterations),memory(d_total_weights),memory(d_region_data),memory(d_ge),memory(d_gmfp),memory(d_gbr1),memory(d_gbr2),memory(d_cohe),memory(d_pmax),memory(d_rayleigh_data),memory(d_i_array)});
  source_bind_t source_meta{h_source.n,h_source.source_point.x,h_source.source_point.y,h_source.source_point.z,h_source.rectangle_z,h_source.rectangle_min.x,h_source.rectangle_min.y,h_source.rectangle_max.x,h_source.rectangle_max.y,h_source.rectangle_size.x,h_source.rectangle_size.y,h_source.rectangle_area};
  phantom_bind_t phantom_meta{h_phantom.N.x,h_phantom.N.y,h_phantom.N.z};
  detector_bind_t detector_meta{h_detector.center.x,h_detector.center.y,h_detector.center.z,h_detector.d.x,h_detector.d.y,h_detector.N.x,h_detector.N.y};
  bindMetadata("egs_bind_source",source_meta,{memory(h_source.xi),memory(h_source.wi),memory(h_source.bin)});
  bindMetadata("egs_bind_phantom",phantom_meta,{memory(h_phantom.x_bounds),memory(h_phantom.y_bounds),memory(h_phantom.z_bounds)});
  bindMetadata("egs_bind_detector",detector_meta);
  for(int b=0;b<SIMULATION_NUM_BLOCKS;++b)for(int c=0;c<NUM_DETECTOR_CAT;++c)
    submitKernel("egs_bind_scores",dim3(1),dim3(1),{memory(d_detector_scores_count[b][c]),memory(d_detector_scores_energy[b][c]),scalar(b),scalar(c)});
  for(int c=0;c<NUM_DETECTOR_CAT;++c)
    submitKernel("egs_bind_totals",dim3(1),dim3(1),{memory(d_detector_totals_count[c]),memory(d_detector_totals_energy[c]),scalar(c)});
#ifdef HECBENCH_EGS_DIAGNOSTIC
  unsigned long* probe=nullptr;allocateBuffer(&probe,30*sizeof(unsigned long));
  submitKernel("egs_probe_bindings",dim3(1),dim3(1),{memory(probe),memory(d_MT_tables),memory(d_MT_params),memory(d_MT_statuses),memory(d_stack.a),memory(d_stack.b),memory(d_stack.c),memory(d_total_step_counts),memory(d_total_list_depth),memory(d_total_num_inner_iterations),memory(d_total_weights),memory(d_region_data),memory(d_ge),memory(d_gmfp),memory(d_gbr1),memory(d_gbr2),memory(d_cohe),memory(d_pmax),memory(d_rayleigh_data),memory(d_i_array)});
  unsigned long results[30];downloadBuffer(results,probe,sizeof(results));
  std::printf("Host ABI source=%zu detector=%zu phantom=%zu\n",sizeof(source_t),sizeof(detector_t),sizeof(phantom_t));
  for(int i=0;i<30;++i)std::printf("binding_probe[%d]=%lu\n",i,results[i]);std::fflush(stdout);
  releaseBuffer(probe);
#endif
}
