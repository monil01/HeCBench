typedef unsigned int uint32_t;
struct Geometry {unsigned int x,y,z;};
struct Geometry blockIdx,blockDim,threadIdx;
float sqrtf(float);float logf(float);float expf(float);float powf(float,float);float rsqrtf(float);float abs(float);
double sqrt(double);double pow(double,double);double trunc(double);

void wsm (
        float * th,
  const float * pii,
        float * q,
        float * qc,
        float * qi,
        float * qr,
        float * qs,
  const float * den,
  const float * p,
  const float * delz,
        float * rain,
        float * rainncv,
        float * sr,
        float * snow,
        float * snowncv,
  const float delt,
  const int ids, const int ide,
  const int jds, const int jde,
  const int kds, const int kde,
  const int ims, const int ime,
  const int jms, const int jme,
  const int kms, const int kme,
  const int ips, const int ipe,
  const int jps, const int jpe,
  const int kps, const int kpe )
{
  float xlf, xmi, acrfac, vt2i, vt2s, supice, diameter ;
  float roqi0, xni0, qimax, value, source, factor, xlwork2 ;
  float t_k, q_k, qr_k, qc_k, qs_k, qi_k, qs1_k, qs2_k, cpm_k, xl_k, w1_k, w2_k, w3_k ;
  float ttp ;
  float dldt ;
  float xa ;
  float xb ;
  float dldti ;
  float xai ;
  float xbi ;
  float qs1[4] ;
  float qs2[4] ;
  float rh1[4] ;
  float rh2[4] ;
  if ( (((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims))) < ide-ids+1 && (((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms))) < jde-jds+1 ) {
    int k ;
   double cpv = 4.*0.461600006103516E+03 ;
   double cp = 7.*287./2. ;
   double cv = cp-287. ;
   double cpd = cp ;
   double pvtr = 0.841900000000000E+03*0.178173289058329E+02/6. ;
   double pvts = 0.117200000000000E+02*0.102654190601850E+02/6. ;
   double xlv1 = 0.419000000000000E+04 - cv ;
   double rslopermax = 1./0.800000000000000E+05 ;
   double rslopesmax = .10000000000000001e-04 ;
   double rsloperbmax = 0.11954406247375457E-03 ;
   double rslopesbmax = .89125093813374589e-02 ;
   double rsloper2max = rslopermax * rslopermax ;
   double rslopes2max = rslopesmax * rslopesmax ;
   double rsloper3max = rsloper2max * rslopermax ;
   double rslopes3max = rslopes2max * rslopesmax ;
   double pidn0r = 0.314159265358979E+01*0.100000000000000E+04*0.800000000000000E+07 ;
   double pidn0s = 0.314159265358979E+01*100.*0.200000000000000E+07 ;
   double precs1 = 4.*0.200000000000000E+07*.65 ;
   double precs2 = 4.*0.200000000000000E+07*.44*sqrt(0.117200000000000E+02)*1.550308 ;
   double qc0 = 4./3.*0.314159265358979E+01*0.100000000000000E+04*(0.800000000000000E-05*0.800000000000000E-05*0.800000000000000E-05)*0.300000000000000E+09/0.127999997138977E+01 ;
   double qck1 = .104*9.8*0.550000000000000E+00/pow((0.300000000000000E+09*0.100000000000000E+04),(1./3.))/0.171800000000000E-04*pow(0.127999997138977E+01,(4./3.)) ;
   double precr1 = 2.*0.314159265358979E+01*0.800000000000000E+07*.78 ;
   double precr2 = 2.*0.314159265358979E+01*0.800000000000000E+07*.31*sqrt(0.841900000000000E+03)*0.182658695197891E+01 ;
   double pacrr = 0.314159265358979E+01*0.800000000000000E+07*0.841900000000000E+03*0.469078683336385E+01*.25*0.100000000000000E+01 ;
   double pacrc = 0.314159265358979E+01*0.200000000000000E+07*0.117200000000000E+02*0.301156382231086E+01*.25*0.100000000000000E+01 ;
   double roqimax = 2.08e22*pow(0.500000000000000E-03,8.0) ;
    float t[4] ;
    float cpm[4] ;
    float xl[4] ;
    for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
      t[k] = th[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] * pii[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] ;
    }
    for( k=kps-1 ;k<=kpe-1;k++) {
      if ( qc[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] < 0.f ) { qc[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] = 0.f ; }
      if ( qi[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] < 0.f ) { qi[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] = 0.f ; }
      if ( qr[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] < 0.f ) { qr[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] = 0.f ; }
      if ( qs[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] < 0.f ) { qs[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] = 0.f ; }
    }
    for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
      cpm[k] = (cpd*(1.f-((q[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))])>(0.100000000362749E-14)?(q[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]):(0.100000000362749E-14)))+((q[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))])>(0.100000000362749E-14)?(q[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]):(0.100000000362749E-14))*cpv) ;
      xl[k] = (0.250000000000000E+07 -xlv1*((t[k])-0.273149993896484E+03)) ;
    }
    float dtcldcr = 120.f ;
    int loops = delt/dtcldcr+.5f ;
    loops = ((loops)>(1)?(loops):(1)) ;
    float dtcld = delt/loops ;
    if ( delt <= dtcldcr) dtcld = delt ;
    int loop ;
    for ( loop = 1 ; loop <= loops ; loop++ ) {
      int mstep = 1 ;
      ttp=0.273149993896484E+03 +0.01f ;
      dldt=cpv-0.419000000000000E+04 ;
      xa=-dldt/0.461600006103516E+03 ;
      xb=xa+0.250000000000000E+07/(0.461600006103516E+03*ttp) ;
      dldti=cpv-0.210600000000000E+04 ;
      xai=-dldti/0.461600006103516E+03 ;
      xbi=xai+0.285000000000000E+07/(0.461600006103516E+03*ttp) ;
      float tr, ltr, tt, pp, qq ;
      for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
        pp = p[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] ;
        tt = t[k] ;
        tr = ttp/tt ;
        ltr = logf(tr) ;
        qq=0.610780029296875E+03*expf(ltr*(xa)+xb*(1.f-tr)) ;
        qq=0.621750414371490E+00*qq/(pp-qq) ;
        qs1[k] = ((qq)>(0.100000000362749E-14)?(qq):(0.100000000362749E-14)) ;
        rh1[k] = ((q[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]/qs1[k])>(0.100000000362749E-14)?(q[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]/qs1[k]):(0.100000000362749E-14)) ;
        if( tt < ttp ) {
          qq=0.610780029296875E+03*expf(ltr*(xai)+xbi*(1.f-tr)) ;
        } else {
          qq=0.610780029296875E+03*expf(ltr*(xa)+xb*(1.f-tr)) ;
        }
        qq = 0.621750414371490E+00 * qq / (pp - qq) ;
        qs2[k] = ((qq)>(0.100000000362749E-14)?(qq):(0.100000000362749E-14)) ;
        rh2[k] = ((q[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]/qs2[k])>(0.100000000362749E-14)?(q[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]/qs2[k]):(0.100000000362749E-14)) ;
      }
      float prevp_reg ;
      float psdep_reg ;
      float praut_reg ;
      float psaut_reg ;
      float pracw_reg ;
      float psaci_reg ;
      float psacw_reg ;
      float pigen_reg ;
      float pidep_reg ;
      float pcond_reg ;
      float psmlt_reg ;
      float psevp_reg ;
      float xni[4] ;
      for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
        xni[k] = 1.e3f ;
      }
      float rsloper[4] ;
      float rslopebr[4] ;
      float rslope2r[4] ;
      float rslope3r[4] ;
      float rslopes[4] ;
      float rslopebs[4] ;
      float rslope2s[4] ;
      float rslope3s[4] ;
      float denfac[4] ;
      float n0sfac[4] ;
      float w1[4] ;
      float w2[4] ;
      float w3[4] ;
      float w ;
      float rmstep ;
      int numdt ;
      for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
        float supcol = 0.273149993896484E+03 - t[k] ;
        n0sfac[k] = ((((expf(0.120000000000000E+00*supcol))<(0.100000000000000E+12/0.200000000000000E+07)?(expf(0.120000000000000E+00*supcol)):(0.100000000000000E+12/0.200000000000000E+07)))>(1.f)?(((expf(0.120000000000000E+00*supcol))<(0.100000000000000E+12/0.200000000000000E+07)?(expf(0.120000000000000E+00*supcol)):(0.100000000000000E+12/0.200000000000000E+07))):(1.f)) ;
        if ( qr[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] <= 0.100000000000000E-08 ) {
          rsloper[k] = rslopermax ;
          rslopebr[k] = rsloperbmax ;
          rslope2r[k] = rsloper2max ;
          rslope3r[k] = rsloper3max ;
        } else {
          rsloper[k] = 1.f/sqrtf(sqrtf(pidn0r/((qr[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))])*(den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))])))) ;
          rslopebr[k] = expf(logf(rsloper[k])*0.800000000000000E+00) ;
          rslope2r[k] = rsloper[k] * rsloper[k] ;
          rslope3r[k] = rslope2r[k] * rsloper[k] ;
        }
        if ( qs[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] <= 0.100000000000000E-08 ) {
          rslopes[k] = rslopesmax ;
          rslopebs[k] = rslopesbmax ;
          rslope2s[k] = rslopes2max ;
          rslope3s[k] = rslopes3max ;
        } else {
          rslopes[k] = 1.f/sqrtf(sqrtf(pidn0s*(n0sfac[k])/((qs[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))])*(den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))])))) ;
          rslopebs[k] = expf(logf(rslopes[k])*0.410000000000000E+00) ;
          rslope2s[k] = rslopes[k] * rslopes[k] ;
          rslope3s[k] = rslope2s[k] * rslopes[k] ;
        }
        denfac[k] = sqrtf(0.127999997138977E+01/den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]) ;
        w1[k] = pvtr*rslopebr[k]*denfac[k]/delz[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] ;
        w2[k] = pvts*rslopebs[k]*denfac[k]/delz[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] ;
        w = ((w1[k])>(w2[k])?(w1[k]):(w2[k])) ;
        numdt = (((int)trunc(w*dtcld+.5f+.5f))>(1)?((int)trunc(w*dtcld+.5f+.5f)):(1)) ;
        if ( numdt >= mstep ) mstep = numdt ;
        float temp = (den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]*((qi[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))])>(0.100000000362749E-14)?(qi[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]):(0.100000000362749E-14))) ;
        temp = sqrtf(sqrtf(temp*temp*temp)) ;
        xni[k] = ((((5.38e7f*temp)>(1.e3f)?(5.38e7f*temp):(1.e3f)))<(1.e6f)?(((5.38e7f*temp)>(1.e3f)?(5.38e7f*temp):(1.e3f))):(1.e6f)) ;
      }
      rmstep = 1.f/mstep ;
      int n ;
      float dtcldden, coeres, rdelz ;
      float den_k, falk1_k, falk1_kp1, fall1_k, delz_k, delz_kp1 ;
      float falk2_k, falk2_kp1, fall2_k;
      for ( n = 1 ; n <= mstep ; n++ ) {
        k = kpe - 1 ;
        den_k = den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] ;
        falk1_kp1 = den_k*qr[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]*w1[k]*rmstep ;
        falk2_kp1 = den_k*qs[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]*w2[k]*rmstep ;
        dtcldden = dtcld/den_k ;
        qr[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] = ((qr[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]-falk1_kp1*dtcldden)>(0.f)?(qr[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]-falk1_kp1*dtcldden):(0.f)) ;
        qs[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] = ((qs[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]-falk2_kp1*dtcldden)>(0.f)?(qs[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]-falk2_kp1*dtcldden):(0.f)) ;
        delz_kp1 = delz[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] ;
        for ( k = kpe-2 ; k >= kps-1 ; k-- ) {
          den_k = den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] ;
          falk1_k = den_k*qr[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]*w1[k]*rmstep ;
          fall1_k = falk1_k ;
          falk2_k = den_k*qs[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]*w2[k]*rmstep ;
          fall2_k = falk2_k ;
          dtcldden = dtcld/den_k ;
          delz_k = delz[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] ;
          rdelz = 1.f/delz_k ;
          qr[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] = ((qr[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]- (falk1_k-falk1_kp1*delz_kp1*rdelz)* dtcldden)>(0.f)?(qr[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]- (falk1_k-falk1_kp1*delz_kp1*rdelz)* dtcldden):(0.f)) ;
          qs[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] = ((qs[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]- (falk2_k-falk2_kp1*delz_kp1*rdelz)* dtcldden)>(0.f)?(qs[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]- (falk2_k-falk2_kp1*delz_kp1*rdelz)* dtcldden):(0.f)) ;
          delz_kp1 = delz_k ;
          falk1_kp1 = falk1_k ;
          falk2_kp1 = falk2_k ;
        }
        for ( k = kpe-1 ; k >= kps-1 ; k-- ) {
          if ( t[k] > 0.273149993896484E+03 && qs[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] > 0.f) {
            xlf = 0.350000000000000E+06 ;
            w3[k] = (expf(logf(((1.496e-6f * (((t[k]))*sqrtf((t[k]))) /(((t[k]))+120.f)/((den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))])))/(8.794e-5f * expf(logf((t[k]))*(1.81f)) / ((p[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))])))))*((.3333333f)))*rsqrtf((1.496e-6f * (((t[k]))*sqrtf((t[k]))) /(((t[k]))+120.f)/((den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]))))*sqrtf(sqrtf(0.127999997138977E+01/(den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))])))) ;
            coeres = rslope2s[k]*sqrtf(rslopes[k]*rslopebs[2]) ;
            psmlt_reg = (1.414e3f*(1.496e-6f * (((t[k]))*sqrtf((t[k]))) /(((t[k]))+120.f)/((den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))])))*(den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]))/xlf*(0.273149993896484E+03 -t[k])*0.314159265358979E+01/2.f
              *n0sfac[k]*(precs1*rslope2s[k]+precs2
                  *w3[k]*coeres) ;
            psmlt_reg = ((((psmlt_reg*dtcld*rmstep)>(-qs[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]*rmstep)?(psmlt_reg*dtcld*rmstep):(-qs[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]*rmstep)))<(0.f)?(((psmlt_reg*dtcld*rmstep)>(-qs[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]*rmstep)?(psmlt_reg*dtcld*rmstep):(-qs[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]*rmstep))):(0.f)) ;
            qs[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] += psmlt_reg ;
            qr[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] -= psmlt_reg ;
            t[k] += xlf/(cpd*(1.f-((q[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))])>(0.100000000362749E-14)?(q[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]):(0.100000000362749E-14)))+((q[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))])>(0.100000000362749E-14)?(q[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]):(0.100000000362749E-14))*cpv)*psmlt_reg ;
          }
        }
      }
      mstep = 1 ;
      numdt = 1 ;
      for ( k = kpe-1 ; k >= kps-1 ; k-- ) {
        if (qi[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] <= 0.f) {
          w2[k] = 0.f ;
        } else {
          xmi = den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]*qi[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]/xni[k] ;
          diameter = ((((0.119000000000000E+02 * sqrtf(xmi))<(0.500000000000000E-03)?(0.119000000000000E+02 * sqrtf(xmi)):(0.500000000000000E-03)))>(1.e-25f)?(((0.119000000000000E+02 * sqrtf(xmi))<(0.500000000000000E-03)?(0.119000000000000E+02 * sqrtf(xmi)):(0.500000000000000E-03))):(1.e-25f)) ;
          w1[k] = 1.49e4f*expf(logf(diameter)*(1.31f)) ;
          w2[k] = w1[k]/delz[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] ;
        }
        numdt = (((int) trunc(w2[k]*dtcld+.5f+.5f))>(1)?((int) trunc(w2[k]*dtcld+.5f+.5f)):(1)) ;
        if(numdt > mstep) mstep = numdt ;
      }
      rmstep = 1.f/mstep ;
      float falkc_k, falkc_kp1, fallc_k = 0.f, fallc_kp1 = 0.f ;
      for ( n = 1 ; n <= mstep ; n++ ) {
        k = kpe - 1 ;
        den_k = den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] ;
        falkc_kp1 = den_k*qi[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]*w2[k]*rmstep ;
        fallc_kp1 = fallc_kp1+falkc_kp1 ;
        qi[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] = ((qi[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]-falkc_kp1*dtcld/den_k)>(0.f)?(qi[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]-falkc_kp1*dtcld/den_k):(0.f)) ;
        delz_kp1 = delz[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] ;
        for ( k = kpe-2 ; k >= kps-1 ; k-- ) {
          den_k = den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] ;
          falkc_k = den_k*qi[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]*w2[k]*rmstep ;
          fallc_k = fallc_k+falkc_k ;
          delz_k = delz[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] ;
          qi[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] = ((qi[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]-(falkc_k-falkc_kp1 *delz_kp1/delz_k)*dtcld/den_k)>(0.f)?(qi[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]-(falkc_k-falkc_kp1 *delz_kp1/delz_k)*dtcld/den_k):(0.f)) ;
          delz_kp1 = delz_k ;
          falkc_kp1 = falkc_k ;
          fallc_kp1 = fallc_k ;
        }
      }
      float fallsum = fall1_k+fall2_k+fallc_k ;
      float fallsum_qsi = fall2_k+fallc_k ;
      rainncv[((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)))] = 0.f ;
      if(fallsum > 0.f) {
        rainncv[((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)))] = fallsum*delz[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((1)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]/0.100000000000000E+04*dtcld*1000.f ;
        rain[((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)))] = fallsum*delz[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((1)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]/0.100000000000000E+04*dtcld*1000.f + rain[((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)))] ;
      }
      snowncv[((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)))] = 0.f ;
      if(fallsum_qsi > 0.f) {
        snowncv[((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)))] = fallsum_qsi*delz[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((0)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]/0.100000000000000E+04*dtcld*1000.f ;
        snow[((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)))] = fallsum_qsi*delz[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((0)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]/0.100000000000000E+04*dtcld*1000.f + snow[((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)))] ;
      }
      sr[((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)))] = 0.f ;
      if ( fallsum > 0.f ) sr[((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)))] = fallsum_qsi*delz[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((0)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]/0.100000000000000E+04*dtcld*1000.f/(rainncv[((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)))]+1.e-12f) ;
      for ( k = kps-1 ; k <= kpe-1 ; k++ ) {
        prevp_reg = 0.f ;
        psdep_reg = 0.f ;
        praut_reg = 0.f ;
        psaut_reg = 0.f ;
        pracw_reg = 0.f ;
        psaci_reg = 0.f ;
        psacw_reg = 0.f ;
        pigen_reg = 0.f ;
        pidep_reg = 0.f ;
        pcond_reg = 0.f ;
        psevp_reg = 0.f ;
        q_k = q[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] ;
        t_k = t[k] ;
        qr_k = qr[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] ;
        qc_k = qc[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] ;
        qs_k = qs[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] ;
        qi_k = qi[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] ;
        qs1_k = qs1[k] ;
        qs2_k = qs2[k] ;
        cpm_k = cpm[k] ;
        xl_k = xl[k] ;
        float supcol = 0.273149993896484E+03 -t_k ;
        xlf = 0.285000000000000E+07 -xl_k ;
        if( supcol < 0.f ) xlf = 0.350000000000000E+06 ;
        if( supcol < 0.f && qi_k > 0.f ) {
          qc_k = qc_k + qi_k ;
          t_k = t_k - xlf/cpm_k*qi_k ;
          qi_k = 0.f ;
        }
        if( supcol > 40.f && qc_k > 0.f ) {
          qi_k = qi_k + qc_k ;
          t_k = t_k + xlf/cpm_k*qc_k ;
          qc_k = 0.f ;
        }
        if ( supcol > 0.f && qc_k > 0.f) {
          float pfrzdtc = ((0.100000000000000E+03*(expf(0.660000000000000E+00*supcol)-1.f) *den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]/0.100000000000000E+04/0.300000000000000E+09*qc_k*qc_k*dtcld)<(qc_k)?(0.100000000000000E+03*(expf(0.660000000000000E+00*supcol)-1.f) *den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]/0.100000000000000E+04/0.300000000000000E+09*qc_k*qc_k*dtcld):(qc_k)) ;
          qi_k = qi_k + pfrzdtc ;
          t_k = t_k + xlf/cpm_k*pfrzdtc ;
          qc_k = qc_k-pfrzdtc ;
        }
        if( supcol > 0.f && qr_k > 0.f ) {
          float temp = rsloper[k] ;
          temp = temp*temp*temp*temp*temp*temp*temp ;
          float pfrzdtr = ((20.f*(0.314159265358979E+01*0.314159265358979E+01)*0.100000000000000E+03*0.800000000000000E+07*0.100000000000000E+04/den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] *(expf(0.660000000000000E+00*supcol)-1.f)*temp*dtcld)<(qr_k)?(20.f*(0.314159265358979E+01*0.314159265358979E+01)*0.100000000000000E+03*0.800000000000000E+07*0.100000000000000E+04/den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] *(expf(0.660000000000000E+00*supcol)-1.f)*temp*dtcld):(qr_k)) ;
          qs_k = qs_k + pfrzdtr ;
          t_k = t_k + xlf/cpm_k*pfrzdtr ;
          qr_k = qr_k-pfrzdtr ;
        }
        n0sfac[k] = ((((expf(0.120000000000000E+00*supcol))<(0.100000000000000E+12/0.200000000000000E+07)?(expf(0.120000000000000E+00*supcol)):(0.100000000000000E+12/0.200000000000000E+07)))>(1.f)?(((expf(0.120000000000000E+00*supcol))<(0.100000000000000E+12/0.200000000000000E+07)?(expf(0.120000000000000E+00*supcol)):(0.100000000000000E+12/0.200000000000000E+07))):(1.f)) ;
        if ( qr_k <= 0.100000000000000E-08 ) {
          rsloper[k] = rslopermax ;
          rslopebr[k] = rsloperbmax ;
          rslope2r[k] = rsloper2max ;
          rslope3r[k] = rsloper3max ;
        } else {
          rsloper[k] = 1.f/(sqrtf(sqrtf(pidn0r/((qr_k)*(den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]))))) ;
          rslopebr[k] = expf(logf(rsloper[k])*0.800000000000000E+00) ;
          rslope2r[k] = rsloper[k] * rsloper[k] ;
          rslope3r[k] = rslope2r[k] * rsloper[k] ;
        }
        if ( qs_k <= 0.100000000000000E-08 ) {
          rslopes[k] = rslopesmax ;
          rslopebs[k] = rslopesbmax ;
          rslope2s[k] = rslopes2max ;
          rslope3s[k] = rslopes3max ;
        } else {
          rslopes[k] = 1.f/(sqrtf(sqrtf(pidn0s*(n0sfac[k])/((qs_k)*(den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]))))) ;
          rslopebs[k] = expf(logf(rslopes[k])*0.410000000000000E+00) ;
          rslope2s[k] = rslopes[k] * rslopes[k] ;
          rslope3s[k] = rslope2s[k] * rslopes[k] ;
        }
        w1_k = ((den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))])*(xl_k)*(xl_k)/((1.414e3f*(1.496e-6f * ((((t_k)))*sqrtf(((t_k)))) /((((t_k)))+120.f)/(((den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]))))*((den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))])))*0.461600006103516E+03*(t_k)*(t_k))+1.f/((qs1_k)*(8.794e-5f * expf(logf((t_k))*(1.81f)) / ((p[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]))))) ;
        w2_k = ((den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))])*(0.285000000000000E+07)*(0.285000000000000E+07)/((1.414e3f*(1.496e-6f * ((((t_k)))*sqrtf(((t_k)))) /((((t_k)))+120.f)/(((den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]))))*((den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))])))*0.461600006103516E+03*(t_k)*(t_k))+1.f/((qs2_k)*(8.794e-5f * expf(logf((t_k))*(1.81f)) / ((p[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]))))) ;
        w3_k = (expf(logf(((1.496e-6f * (((t_k))*sqrtf((t_k))) /(((t_k))+120.f)/((den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))])))/(8.794e-5f * expf(logf((t_k))*(1.81f)) / ((p[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))])))))*((.3333333f)))*rsqrtf((1.496e-6f * (((t_k))*sqrtf((t_k))) /(((t_k))+120.f)/((den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]))))*sqrtf(sqrtf(0.127999997138977E+01/(den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))])))) ;
        float supsat = ((q_k)>(0.100000000362749E-14)?(q_k):(0.100000000362749E-14))-qs1_k ;
        float satdt = supsat/dtcld ;
        if(qc_k > qc0) {
          praut_reg = qck1*expf(logf(qc_k)*((7.f/3.f))) ;
          praut_reg = ((praut_reg)<(qc_k/dtcld)?(praut_reg):(qc_k/dtcld)) ;
        }
        if(qr_k > 0.100000000000000E-08 && qc_k > 0.100000000362749E-14) {
          pracw_reg = ((pacrr*rslope3r[k]*rslopebr[k] *qc_k*denfac[k])<(qc_k/dtcld)?(pacrr*rslope3r[k]*rslopebr[k] *qc_k*denfac[k]):(qc_k/dtcld)) ;
        }
        if(qr_k > 0.f) {
          coeres = rslope2r[k]*sqrtf(rsloper[k]*rslopebr[k]) ;
          prevp_reg = (rh1[k]-1.f)*(precr1*rslope2r[k]
              +precr2*w3_k*coeres)/w1_k ;
          if(prevp_reg < 0.f) {
            prevp_reg = ((prevp_reg)>(-qr_k/dtcld)?(prevp_reg):(-qr_k/dtcld)) ;
            prevp_reg = ((prevp_reg)>(satdt/2.f)?(prevp_reg):(satdt/2.f)) ;
          } else {
            prevp_reg = ((prevp_reg)<(satdt/2.f)?(prevp_reg):(satdt/2.f)) ;
          }
        }
        float rdtcld = 1.f/dtcld ;
        supsat = ((q_k)>(0.100000000362749E-14)?(q_k):(0.100000000362749E-14))-qs2_k ;
        satdt = supsat/dtcld ;
        int ifsat = 0 ;
        float temp = (den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]*((qi_k)>(0.100000000362749E-14)?(qi_k):(0.100000000362749E-14))) ;
        temp = sqrtf(sqrtf(temp*temp*temp)) ;
        xni[k] = ((((5.38e7f*temp)>(1.e3f)?(5.38e7f*temp):(1.e3f)))<(1.e6f)?(((5.38e7f*temp)>(1.e3f)?(5.38e7f*temp):(1.e3f))):(1.e6f)) ;
        float eacrs = expf(0.07f*(-supcol)) ;
        if(qs_k > 0.100000000000000E-08 && qc_k > 0.100000000362749E-14) {
          psacw_reg = ((pacrc*n0sfac[k]*rslope3s[k] *rslopebs[k]*qc_k*denfac[k])<(qc_k*rdtcld)?(pacrc*n0sfac[k]*rslope3s[k] *rslopebs[k]*qc_k*denfac[k]):(qc_k*rdtcld)) ;
        }
        if(supcol > 0) {
          if(qs_k > 0.100000000000000E-08 && qi_k > 0.100000000362749E-14) {
            xmi = den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]*qi_k/xni[k] ;
            diameter = ((0.119000000000000E+02 * sqrtf(xmi))<(0.500000000000000E-03)?(0.119000000000000E+02 * sqrtf(xmi)):(0.500000000000000E-03)) ;
            vt2i = 1.49e4f*powf(diameter,1.31f) ;
            vt2s = pvts*rslopebs[k]*denfac[k] ;
            acrfac = 2.f*rslope3s[k]+2.f*diameter*rslope2s[k]
              +diameter*diameter*rslopes[k] ;
            psaci_reg = 0.314159265358979E+01*qi_k*eacrs*0.200000000000000E+07*n0sfac[k]
              *abs(vt2s-vt2i)*acrfac*.25f ;
          }
          if(qi_k > 0 && ifsat != 1) {
            xmi = den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]*qi_k/xni[k] ;
            diameter = 0.119000000000000E+02 * sqrtf(xmi) ;
            pidep_reg = 4.f*diameter*xni[k]*(rh2[k]-1.f)/w2_k ;
            supice = satdt-prevp_reg ;
            if(pidep_reg < 0.f) {
              pidep_reg = ((((pidep_reg)>(satdt*.5f)?(pidep_reg):(satdt*.5f)))>(supice)?(((pidep_reg)>(satdt*.5f)?(pidep_reg):(satdt*.5f))):(supice)) ;
              pidep_reg = ((pidep_reg)>(-qi_k*rdtcld)?(pidep_reg):(-qi_k*rdtcld)) ;
            } else {
              pidep_reg = ((((pidep_reg)<(satdt*.5f)?(pidep_reg):(satdt*.5f)))<(supice)?(((pidep_reg)<(satdt*.5f)?(pidep_reg):(satdt*.5f))):(supice)) ;
            }
            if(abs(prevp_reg+pidep_reg) >= abs(satdt)) ifsat = 1 ;
          }
          if( qs_k > 0.f && ifsat != 1) {
            coeres = rslope2s[k]*sqrtf(rslopes[k]*rslopebs[k]) ;
            psdep_reg = (rh2[k]-1.f)*n0sfac[k]
              *(precs1*rslope2s[k]+precs2
                  *w3_k*coeres)/w2_k ;
            supice = satdt-prevp_reg-pidep_reg ;
            if(psdep_reg < 0.f) {
              psdep_reg = ((psdep_reg)>(-qs_k*rdtcld)?(psdep_reg):(-qs_k*rdtcld)) ;
              psdep_reg = ((((psdep_reg)>(satdt*.5f)?(psdep_reg):(satdt*.5f)))>(supice)?(((psdep_reg)>(satdt*.5f)?(psdep_reg):(satdt*.5f))):(supice)) ;
            } else {
              psdep_reg = ((((psdep_reg)<(satdt*.5f)?(psdep_reg):(satdt*.5f)))<(supice)?(((psdep_reg)<(satdt*.5f)?(psdep_reg):(satdt*.5f))):(supice)) ;
            }
            if(abs(prevp_reg+pidep_reg+psdep_reg) >= abs(satdt))
              ifsat = 1 ;
          }
          if(supsat > 0 && ifsat != 1) {
            supice = satdt-prevp_reg-pidep_reg-psdep_reg ;
            xni0 = 1.e3f*expf(0.1f*supcol) ;
            roqi0 = 4.92e-11f*expf(logf(xni0)*(1.33f));
            pigen_reg = ((0.f)>((roqi0/den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]-((qi_k)>(0.f)?(qi_k):(0.f))) *rdtcld)?(0.f):((roqi0/den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))]-((qi_k)>(0.f)?(qi_k):(0.f))) *rdtcld)) ;
            pigen_reg = ((((pigen_reg)<(satdt)?(pigen_reg):(satdt)))<(supice)?(((pigen_reg)<(satdt)?(pigen_reg):(satdt))):(supice)) ;
          }
          if(qi_k > 0.f) {
            qimax = roqimax/den[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] ;
            psaut_reg = ((0.f)>((qi_k-qimax)*rdtcld)?(0.f):((qi_k-qimax)*rdtcld)) ;
          }
        }
        if(supcol < 0.f) {
          if(qs_k > 0.f && rh1[k] < 1.f) {
            psevp_reg = psdep_reg*w2_k/w1_k ;
          }
          psevp_reg = ((((psevp_reg)>(-qs_k*rdtcld)?(psevp_reg):(-qs_k*rdtcld)))<(0.f)?(((psevp_reg)>(-qs_k*rdtcld)?(psevp_reg):(-qs_k*rdtcld))):(0.f)) ;
        }
        if(t_k<=0.273149993896484E+03) {
          value = ((0.100000000362749E-14)>(qc_k)?(0.100000000362749E-14):(qc_k)) ;
          source = (praut_reg+pracw_reg+psacw_reg)*dtcld ;
          if (source > value) {
            factor = value/source ;
            praut_reg = praut_reg*factor ;
            pracw_reg = pracw_reg*factor ;
            psacw_reg = psacw_reg*factor ;
          }
          value = ((0.100000000362749E-14)>(qi_k)?(0.100000000362749E-14):(qi_k)) ;
          source = (psaut_reg+psaci_reg-pigen_reg-pidep_reg)*dtcld ;
          if (source > value) {
            factor = value/source ;
            psaut_reg = psaut_reg*factor ;
            psaci_reg = psaci_reg*factor ;
            pigen_reg = pigen_reg*factor ;
            pidep_reg = pidep_reg*factor ;
          }
          value = ((0.100000000362749E-14)>(qr_k)?(0.100000000362749E-14):(qr_k)) ;
          source = (-praut_reg+pracw_reg-prevp_reg)*dtcld ;
          if (source > value) {
            factor = value/source ;
            praut_reg = praut_reg*factor ;
            pracw_reg = pracw_reg*factor ;
            prevp_reg = prevp_reg*factor ;
          }
          value = ((0.100000000362749E-14)>(qs_k)?(0.100000000362749E-14):(qs_k)) ;
          source = (-psdep_reg+psaut_reg-psaci_reg-psacw_reg)*dtcld ;
          if (source > value) {
            factor = value/source ;
            psdep_reg = psdep_reg*factor ;
            psaut_reg = psaut_reg*factor ;
            psaci_reg = psaci_reg*factor ;
            psacw_reg = psacw_reg*factor ;
          }
          w3_k=-(prevp_reg+psdep_reg+pigen_reg+pidep_reg) ;
          q_k = q_k+w3_k*dtcld ;
          qc_k = ((qc_k-(praut_reg+pracw_reg+psacw_reg)*dtcld)>(0.f)?(qc_k-(praut_reg+pracw_reg+psacw_reg)*dtcld):(0.f)) ;
          qr_k = ((qr_k+(praut_reg+pracw_reg+prevp_reg)*dtcld)>(0.f)?(qr_k+(praut_reg+pracw_reg+prevp_reg)*dtcld):(0.f)) ;
          qi_k = ((qi_k-(psaut_reg+psaci_reg-pigen_reg-pidep_reg)*dtcld)>(0.f)?(qi_k-(psaut_reg+psaci_reg-pigen_reg-pidep_reg)*dtcld):(0.f)) ;
          qs_k = ((qs_k+(psdep_reg+psaut_reg+psaci_reg+psacw_reg)*dtcld)>(0.f)?(qs_k+(psdep_reg+psaut_reg+psaci_reg+psacw_reg)*dtcld):(0.f)) ;
          xlf = 0.285000000000000E+07 -xl_k ;
          xlwork2 = -0.285000000000000E+07*(psdep_reg+pidep_reg+pigen_reg)-xl_k*prevp_reg-xlf*psacw_reg ;
          t_k = t_k-xlwork2/cpm_k*dtcld ;
        } else {
          value = ((0.100000000362749E-14)>(qc_k)?(0.100000000362749E-14):(qc_k)) ;
          source=(praut_reg+pracw_reg+psacw_reg)*dtcld ;
          if (source > value) {
            factor = value/source ;
            praut_reg = praut_reg*factor ;
            pracw_reg = pracw_reg*factor ;
            psacw_reg = psacw_reg*factor ;
          }
          value = ((0.100000000362749E-14)>(qr_k)?(0.100000000362749E-14):(qr_k)) ;
          source = (-praut_reg-pracw_reg-prevp_reg-psacw_reg)*dtcld ;
          if (source > value) {
            factor = value/source ;
            praut_reg = praut_reg*factor ;
            pracw_reg = pracw_reg*factor ;
            prevp_reg = prevp_reg*factor ;
            psacw_reg = psacw_reg*factor ;
          }
          value = ((0.100000000000000E-08)>(qs_k)?(0.100000000000000E-08):(qs_k)) ;
          source=(-psevp_reg)*dtcld ;
          if (source > value) {
            factor = value/source ;
            psevp_reg = psevp_reg*factor ;
          }
          w3_k=-(prevp_reg+psevp_reg) ;
          q_k = q_k+w3_k*dtcld ;
          qc_k = ((qc_k-(praut_reg+pracw_reg+psacw_reg)*dtcld)>(0.f)?(qc_k-(praut_reg+pracw_reg+psacw_reg)*dtcld):(0.f)) ;
          qr_k = ((qr_k+(praut_reg+pracw_reg+prevp_reg +psacw_reg)*dtcld)>(0.f)?(qr_k+(praut_reg+pracw_reg+prevp_reg +psacw_reg)*dtcld):(0.f)) ;
          qs_k = ((qs_k+psevp_reg*dtcld)>(0.f)?(qs_k+psevp_reg*dtcld):(0.f)) ;
          xlf = 0.285000000000000E+07 -xl_k ;
          xlwork2 = -xl_k*(prevp_reg+psevp_reg) ;
          t_k = t_k-xlwork2/cpm_k*dtcld ;
        }
        cpv = cpv ;
        ttp=0.273149993896484E+03 +0.01f ;
        dldt=cpv-0.419000000000000E+04 ;
        xa=-dldt/0.461600006103516E+03 ;
        xb=xa+0.250000000000000E+07/(0.461600006103516E+03*ttp) ;
        dldti=cpv-0.210600000000000E+04 ;
        xai=-dldti/0.461600006103516E+03 ;
        xbi=xai+0.285000000000000E+07/(0.461600006103516E+03*ttp) ;
        tr=ttp/t_k ;
        qs1_k=0.610780029296875E+03*expf(logf(tr)*(xa))*expf(xb*(1.f-tr)) ;
        qs1_k = 0.621750414371490E+00 * qs1_k / (p[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] - qs1_k) ;
        qs1_k = ((qs1_k)>(0.100000000362749E-14)?(qs1_k):(0.100000000362749E-14)) ;
        w1_k = ((((q_k)>(0.100000000362749E-14)?(q_k):(0.100000000362749E-14))-(qs1_k)) /
            (1.f+(xl_k)*(xl_k)/(0.461600006103516E+03*(cpm_k))*(qs1_k)/((t_k)*(t_k)))) ;
        pcond_reg = ((((w1_k/dtcld)>(0.f)?(w1_k/dtcld):(0.f)))<(((q_k)>(0.f)?(q_k):(0.f))/dtcld)?(((w1_k/dtcld)>(0.f)?(w1_k/dtcld):(0.f))):(((q_k)>(0.f)?(q_k):(0.f))/dtcld)) ;
        if(qc_k > 0.f && w1_k < 0.f) {
          pcond_reg = ((w1_k)>(-qc_k)?(w1_k):(-qc_k))/dtcld ;
        }
        q_k = q_k-pcond_reg*dtcld ;
        qc_k = ((qc_k+pcond_reg*dtcld)>(0.f)?(qc_k+pcond_reg*dtcld):(0.f)) ;
        t_k = t_k+pcond_reg*xl_k/cpm_k*dtcld ;
        if(qc_k <= 0.100000000362749E-14) qc_k = 0.f ;
        if(qi_k <= 0.100000000362749E-14) qi_k = 0.f ;
        q[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] = q_k ;
        t[k] = t_k ;
        qr[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] = qr_k ;
        qc[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] = qc_k ;
        qs[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] = qs_k ;
        qi[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] = qi_k ;
        qs1[k] = qs1_k ;
      }
    }
    for ( k = kps-1 ; k <= kpe-1 ; k++ )
      th[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] = t[k] / pii[(((((threadIdx.x)+(blockIdx.x)*(blockDim.x)+(ips)-(ims)))+((k)*(ime-ims+1)))+((((threadIdx.y)+(blockIdx.y)*(blockDim.y)+(jps)-(jms)))*(ime-ims+1)*(kme-kms+1)))] ;
  }
}
