#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
static int randx = 7;
static double drnd()
{
  const int lastrand = randx;
  randx = (1103515245L * randx + 12345) & 0x7FFFFFFF;
  return (double)lastrand / 2147483648.0;
}



int main(int argc,char**argv){
 const int nbodies=atoi(argv[1]); int i;double rsc,vsc,r,v,x,y,z,sq,scale;
 std::vector<float4> posMass(nbodies),accVel(nbodies,make_float4(0,0,0,0));std::vector<float2>vel(nbodies);
  rsc = (3 * 3.1415926535897932384626433832795) / 16;
  vsc = sqrt(1.0 / rsc);
  for (i = 0; i < nbodies; i++) {
    float4 p;
    p.w = 1.f / nbodies;
    r = 1.0 / sqrt(pow(drnd()*0.999, -2.0/3.0) - 1.0);
    do {
      x = drnd()*2.0 - 1.0;
      y = drnd()*2.0 - 1.0;
      z = drnd()*2.0 - 1.0;
      sq = x*x + y*y + z*z;
    } while (sq > 1.0);
    scale = rsc * r / sqrt(sq);
    p.x = x * scale;
    p.y = y * scale;
    p.z = z * scale;
    posMass[i] = p;

    do {
      x = drnd();
      y = drnd() * 0.1;
    } while (y > x*x * pow(1 - x*x, 3.5));
    v = x * sqrt(2.0 / sqrt(1 + r*r));
    do {
      x = drnd()*2.0 - 1.0;
      y = drnd()*2.0 - 1.0;
      z = drnd()*2.0 - 1.0;
      sq = x*x + y*y + z*z;
    } while (sq > 1.0);
    scale = vsc * v / sqrt(sq);
    float2 v;
    v.x = x * scale;
    v.y = y * scale;
    accVel[i].w = z * scale;
    vel[i] = v;
  }


 FILE*f=fopen(argv[2],"wb"); fwrite(posMass.data(),sizeof(float4),nbodies,f);fwrite(vel.data(),sizeof(float2),nbodies,f);fwrite(accVel.data(),sizeof(float4),nbodies,f);fclose(f);
}
