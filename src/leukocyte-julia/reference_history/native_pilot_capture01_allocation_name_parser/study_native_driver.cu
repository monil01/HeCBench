#include "study_capture.h"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <unistd.h>

int leukocyte_legacy_main(int argc,char**argv);
int main(int argc,char**argv) {
  try {
    if(argc<3 || argc>4) throw std::runtime_error("video path, persistent output path, optional pilot required");
    const std::string video=argv[1],folder=argv[2];
    const bool pilot=argc==4 && std::string(argv[3])=="pilot";
    std::filesystem::create_directories(folder);
    auto& state=leukostudy::state();
    for(int trial=0;trial<=(pilot?1:5);trial++) {
      state.prefix.clear(); state.output.clear(); state.correct=true; state.seconds=0;
      auto started=std::chrono::steady_clock::now();
      for(int frames:{1,3,100}) {
        leukostudy::begin_case(folder,"cuda",trial,frames,trial>0);
        std::string count=std::to_string(frames);
        char name[]="leukocyte";
        char* args[]={name,const_cast<char*>(video.c_str()),const_cast<char*>(count.c_str())};
        if(leukocyte_legacy_main(3,args)!=0) throw std::runtime_error("native application returned failure");
        leukostudy::end_case();
      }
      double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
      if(trial==0) continue;
      size_t prefix_count=state.prefix.size();
      state.prefix.insert(state.prefix.end(),state.output.begin(),state.output.end());
      std::string artifact="trial"+std::to_string(trial)+".bin";
      std::ofstream output(artifact,std::ios::binary);
      leukostudy::write_bytes(output,state.prefix.data(),state.prefix.size()*8);
      output.close();
      printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":null,\"max_rel_error\":null,\"input_prefix_values\":%zu,\"instrumented_driver_wall_seconds\":%.17g,\"timing_region\":\"sum_original_3stage_kernel_ROIs_1_3_100frames_including_iteration_counter_store\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",
          trial,state.seconds,state.correct?"true":"false",state.prefix.size(),prefix_count,wall,artifact.c_str(),state.prefix.size());
      fflush(stdout);
      if(!state.correct) return 1;
    }
    return 0;
  } catch(const std::exception& error) { fprintf(stderr,"%s\n",error.what()); return 1; }
}
