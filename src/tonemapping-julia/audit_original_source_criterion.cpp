/**********************************************************************
  Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  •   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  •   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************************************************************/
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <algorithm>
int main(int argc,char**argv){try{for(int arg=1;arg<argc;arg++){std::ifstream f(argv[arg],std::ios::binary);std::vector<float>a(1093708);if(!f.read((char*)a.data(),a.size()*4)||f.peek()!=EOF)throw std::runtime_error("artifact size");int po=0,oo=546868;const unsigned WS[]={16,32,64,256},HS[]={16,16,32,256};printf("{\"artifact\":\"%s\",\"cases\":[",argv[arg]);for(int caseid=0;caseid<4;caseid++){unsigned width=WS[caseid],height=HS[caseid],numChannels=4;int n=width*height*4;float*input=a.data()+po;float*output=a.data()+oo;float*params=input+2*n;float averageLuminance=params[0],gammaPattanaik=params[1],cPattanaik=params[2],deltaPattanaik=params[3];if(params[4]!=width||params[5]!=4||params[6]!=height)throw std::runtime_error("actual dimensions");std::vector<float>ref(n);float*referenceOutput=ref.data();
  float gcPattanaik = cPattanaik * averageLuminance;

  for (unsigned int y = 0; y < height; y++)
  {
    for (unsigned int x = 0; x < width; x++)
    {
      float yLPattanaik = 0.0f;
      float cLPattanaik = 0.0f;

      float r, g, b;
      float r1 = input[y * width * numChannels + (x * numChannels + 0)];
      float g1 = input[y * width * numChannels + (x * numChannels + 1)];
      float b1 = input[y * width * numChannels + (x * numChannels + 2)];

      //Calculating the luminance value
      float yLuminance = (0.2126f * r1) + (0.7152f * g1) + (0.0722f * b1);

      if (x != 0 && y != 0 && x != width - 1 && y != height - 1)
      {
        //Calculating mean
        float leftUp = 0.0f;
        float up = 0.0f;
        float rightUp = 0.0f;
        float left = 0.0f;
        float right = 0.0f;
        float leftDown = 0.0f;
        float down = 0.0f;
        float rightDown = 0.0f;

        r = input[width * numChannels * (y - 1) + ((x - 1) * numChannels) + 0 ];
        g = input[width * numChannels * (y - 1) + ((x - 1) * numChannels) + 1 ];
        b = input[width * numChannels * (y - 1) + ((x - 1) * numChannels) + 2 ];

        leftUp = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);

        r = input[width * numChannels * (y - 1) + ((x) * numChannels) + 0 ];
        g = input[width * numChannels * (y - 1) + ((x) * numChannels) + 1 ];
        b = input[width * numChannels * (y - 1) + ((x) * numChannels) + 2 ];

        up = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);

        r = input[width * numChannels * (y - 1) + ((x + 1) * numChannels) + 0 ];
        g = input[width * numChannels * (y - 1) + ((x + 1) * numChannels) + 1 ];
        b = input[width * numChannels * (y - 1) + ((x + 1) * numChannels) + 2 ];

        rightUp = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);

        r = input[width * numChannels * (y) + ((x - 1) * numChannels) + 0 ];
        g = input[width * numChannels * (y) + ((x - 1) * numChannels) + 1 ];
        b = input[width * numChannels * (y) + ((x - 1) * numChannels) + 2 ];

        left = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);

        r = input[width * numChannels * (y) + ((x + 1) * numChannels) + 0 ];
        g = input[width * numChannels * (y) + ((x + 1) * numChannels) + 1 ];
        b = input[width * numChannels * (y) + ((x + 1) * numChannels) + 2 ];

        right = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);

        r = input[width * numChannels * (y + 1) + ((x - 1) * numChannels) + 0 ];
        g = input[width * numChannels * (y + 1) + ((x - 1) * numChannels) + 1 ];
        b = input[width * numChannels * (y + 1) + ((x - 1) * numChannels) + 2 ];

        leftDown = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);

        r = input[width * numChannels * (y + 1) + ((x) * numChannels) + 0 ];
        g = input[width * numChannels * (y + 1) + ((x) * numChannels) + 1 ];
        b = input[width * numChannels * (y + 1) + ((x) * numChannels) + 2 ];

        down = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);

        r = input[width * numChannels * (y + 1) + ((x + 1) * numChannels) + 0 ];
        g = input[width * numChannels * (y + 1) + ((x + 1) * numChannels) + 1 ];
        b = input[width * numChannels * (y + 1) + ((x + 1) * numChannels) + 2 ];

        rightDown = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);

        //Calculate median

        yLPattanaik = (leftUp + up + rightUp + left + right + leftDown + down +
            rightDown) / 8;
      }
      else
      {
        yLPattanaik = yLuminance;
      }

      cLPattanaik = yLPattanaik * log(deltaPattanaik + yLPattanaik / yLuminance) +
        gcPattanaik;

      float yDPattanaik = yLuminance / (yLuminance + cLPattanaik);

      r = pow((r1 / yLuminance), gammaPattanaik) * yDPattanaik;
      g = pow((g1 / yLuminance), gammaPattanaik) * yDPattanaik;
      b = pow((b1 / yLuminance), gammaPattanaik) * yDPattanaik;

      referenceOutput[width * numChannels * y + (x * numChannels + 0)] = r;
      referenceOutput[width * numChannels * y + (x * numChannels + 1)] = g;
      referenceOutput[width * numChannels * y + (x * numChannels + 2)] = b;
      referenceOutput[width * numChannels * y + (x * numChannels + 3)] =
        input[width * numChannels * y + (x * numChannels + 3)];
    }
  }

  float error = 0.0f;

  for (unsigned int y = 0; y < height; y++)
  {
    for (unsigned int x = 0; x < width; x++)
    {
      error += referenceOutput[width * numChannels * y + (x * numChannels + 0)] -
        output[width * numChannels * y + (x * numChannels + 0)];
      error += referenceOutput[width * numChannels * y + (x * numChannels + 1)] -
        output[width * numChannels * y + (x * numChannels + 1)];
      error += referenceOutput[width * numChannels * y + (x * numChannels + 2)] -
        output[width * numChannels * y + (x * numChannels + 2)];
      error += referenceOutput[width * numChannels * y + (x * numChannels + 3)] -
        output[width * numChannels * y + (x * numChannels + 3)];
    }
  }

  error = error / (height * width);

double maxerror=0;for(int i=0;i<n;i++){if(!std::isfinite(referenceOutput[i])||!std::isfinite(output[i]))throw std::runtime_error("nonfinite");maxerror=std::max(maxerror,double(std::abs(referenceOutput[i]-output[i])));}if(!std::isfinite(error)||error>0.000001f)throw std::runtime_error("original signed source threshold");printf("%s{\"width\":%u,\"height\":%u,\"signed_source_error\":%.12g,\"max_abs_vs_source_CPU\":%.12g,\"original_threshold\":1e-6,\"pass\":true}",caseid?",":"",width,height,double(error),maxerror);po+=2*n+13;oo+=2*n+6;}printf("]}\n");if(po!=546868||oo!=1093708)throw std::runtime_error("offset");}}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}return 0;}