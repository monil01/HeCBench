#include "../../src/heartwall-cuda/main.h"
#include "../../src/heartwall-cuda/util/avi/avilib.h"
#include "../../src/heartwall-cuda/util/avi/avimod.h"
#include "../../src/heartwall-cuda/util/file/file.h"
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cstdlib>
template<class T>void save(const std::string&name,T*data,size_t n){std::ofstream f(name,std::ios::binary);f.write((char*)data,n*sizeof(T));}
int main(int argc,char**argv){std::filesystem::path data=argv[1],output=argv[2];std::filesystem::create_directories(output);params_common common={};common.common_mem=sizeof(common);common.frames_processed=11;auto video=data/"test.avi";auto input=data/"input.txt";avi_t*frames=AVI_open_input_file(video.string().data(),1);if(!frames)return 1;common.no_frames=AVI_video_frames(frames);common.frame_rows=AVI_video_height(frames);common.frame_cols=AVI_video_width(frames);common.frame_elem=common.frame_rows*common.frame_cols;common.frame_mem=sizeof(fp)*common.frame_elem;read_parameters(input.string().c_str(),&common.tSize,&common.sSize,&common.maxMove,&common.alpha);read_header(input.string().c_str(),&common.endoPoints,&common.epiPoints);common.allPoints=common.endoPoints+common.epiPoints;common.endo_mem=common.endoPoints*sizeof(int);common.epi_mem=common.epiPoints*sizeof(int);if(common.tSize!=25||common.sSize!=40||common.maxMove!=10)return 2;
#include "source_geometry_assignments.inc"
std::vector<int>er(common.endoPoints),ec(common.endoPoints),pr(common.epiPoints),pc(common.epiPoints);read_data(input.string().c_str(),common.endoPoints,er.data(),ec.data(),common.epiPoints,pr.data(),pc.data());save((output/"common.bin").string(),&common,1);save((output/"endoRow_i32.bin").string(),er.data(),er.size());save((output/"endoCol_i32.bin").string(),ec.data(),ec.size());save((output/"epiRow_i32.bin").string(),pr.data(),pr.size());save((output/"epiCol_i32.bin").string(),pc.data(),pc.size());for(int frame=0;frame<11;frame++){fp*f=get_frame(frames,frame,0,0,1);save((output/("frame"+std::to_string(frame)+"_f32.bin")).string(),f,common.frame_elem);free(f);}AVI_close(frames);return 0;}
