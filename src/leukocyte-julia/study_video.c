/* Narrow original AVI decoder ABI only; no detection/tracking computation. */
#include "../leukocyte-cuda/avilib.h"
#include <stddef.h>
void*leuko_video_open(char*path){return AVI_open_input_file(path,1);}
int leuko_video_width(void*p){return AVI_video_width((avi_t*)p);}
int leuko_video_height(void*p){return AVI_video_height((avi_t*)p);}
long leuko_video_frames(void*p){return AVI_video_frames((avi_t*)p);}
long leuko_video_read(void*p,long frame,unsigned char*out,long capacity){avi_t*v=(avi_t*)p;int keyframe=0;if(!v||frame<0||frame>=AVI_video_frames(v))return -1;long size=AVI_frame_size(v,frame);if(size<0||size>capacity)return -2;if(AVI_set_video_position(v,frame)<0)return -3;return AVI_read_frame(v,(char*)out,&keyframe);}
int leuko_video_close(void*p){return AVI_close((avi_t*)p);}
