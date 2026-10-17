#define main preserved_native_main
#include "../overlay-cuda/main.cu"
#undef main
#include <vector>
int main(){constexpr int width=640,height=480,pixels=width*height;const int count=int(pixels*0.8f);std::vector<float3>image(pixels);std::vector<Box>boxes(count);srand(123);for(auto&v:image){v.x=rand()%256;v.y=rand()%256;v.z=rand()%256;}for(auto&b:boxes){b.width=64+rand()%128;b.height=64+rand()%128;b.left=rand()%(width-64);b.top=rand()%(height-64);}FILE*f=fopen("study_input_f32.bin","wb");fwrite(image.data(),sizeof(float3),pixels,f);fclose(f);f=fopen("study_boxes_i32.bin","wb");fwrite(boxes.data(),sizeof(Box),count,f);fclose(f);f=fopen("study_shape.json","w");fprintf(f,"{\"width\":640,\"height\":480,\"pixels\":307200,\"boxes\":%d,\"float3_bytes\":%zu,\"Box_bytes\":%zu,\"color\":[255,204,203,1]}\n",count,sizeof(float3),sizeof(Box));fclose(f);printf("Source srand123 image640x480 pixelFloat3bytes=%zu Boxbytes=%zu boxes=%d\n",sizeof(float3),sizeof(Box),count);}
