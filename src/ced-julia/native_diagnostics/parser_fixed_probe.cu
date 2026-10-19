#define main original_ced_main
#include "../native_reference/main_parser_border_fixed.cu"
#undef main
int main(int argc,char**argv){Params p(argc,argv);unsigned char *frames[1];int rows,cols,size;read_input(frames,rows,cols,size,p);FILE*f=fopen(".porting-state/diagnostics/ced/parser_fixed_pixels_u8.bin","wb");fwrite(frames[0],1,size,f);fclose(f);free(frames[0]);return 0;}
