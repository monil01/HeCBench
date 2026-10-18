/*****************************************************************************
 * MULTIANS source preparation reused under LGPL-3.0, 2017-2019 André Weißenberger.
 * Independent scalar decoder below checks the actual compressed stream/table.
 *****************************************************************************/
#include "ans_encoder.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <iomanip>
static uint32_t takebit(const uint32_t*in,size_t&at){uint32_t b=(in[at/32]>>(at%32))&1;at++;return b;}
int main(int argc,char**argv){const size_t N=1048576;std::string out=argc>1?argv[1]:".";std::ofstream data(out+"/input_u32.bin",std::ios::binary),oracle(out+"/oracle_u8.bin",std::ios::binary),meta(out+"/source_cases.tsv");size_t offset=0,oo=0;int id=0;
for(float lambda=.1f;lambda<2.5f;lambda+=.16){auto dist=ANSTableGenerator::generate_distribution(5,256,1024,[&](double x){return lambda*exp(-lambda*x);});auto random=ANSTableGenerator::generate_test_data(dist.dist,N,1024,5);auto table=ANSTableGenerator::generate_table(dist.prob,dist.dist,nullptr,256,1024);auto enc=ANSTableGenerator::generate_encoder_table(table);auto dec=ANSTableGenerator::get_decoder_table(enc);auto compressed=ANSEncoder::encode(random->data(),N,enc);size_t nw=compressed->get_compressed_size(),ns=(nw+3)/4,blocks=(ns+127)/128;std::vector<uint32_t>words(nw+32);memcpy(words.data(),compressed->get_compressed_data(),(nw+4)*4);std::vector<uint32_t>packed(dec->get_size());static_assert(sizeof(CUHDCodetableItem)==4,"tableABI");memcpy(packed.data(),dec->get(),packed.size()*4);
// Independent scalar bitreader, no slidingwindow/parallel synchronization algorithm.
size_t bit=32-compressed->get_first_bit();uint32_t state=compressed->get_first_state();for(size_t j=0;j<N;j++){if(state<1024||state>=2048)throw std::runtime_error("state range");auto hit=packed[state-1024];uint8_t symbol=(hit>>16)&255;if(symbol!=(*random)[N-1-j])throw std::runtime_error("serial roundtrip");uint32_t count=hit>>24,low=0;for(uint32_t k=0;k<count;k++)low|=takebit(words.data(),bit)<<k;state=((hit&65535)<<count)|low;while(state<1024)state=(state<<1)|takebit(words.data(),bit);}
uint32_t lambda_bits;memcpy(&lambda_bits,&lambda,4);meta<<id<<'\t'<<lambda_bits<<'\t'<<N<<'\t'<<nw<<'\t'<<ns<<'\t'<<blocks<<'\t'<<compressed->get_first_state()<<'\t'<<compressed->get_first_bit()<<'\t'<<offset<<'\t'<<(nw+32)<<'\t'<<(offset+words.size())<<'\t'<<packed.size()<<'\t'<<oo<<'\t'<<bit<<'\n';data.write((char*)words.data(),words.size()*4);data.write((char*)packed.data(),packed.size()*4);offset+=words.size()+packed.size();std::reverse(random->begin(),random->begin()+N);oracle.write((char*)random->data(),N);oo+=N;std::cout<<"SOURCE_SERIAL_ROUNDTRIP "<<id<<" lambda "<<lambda<<" units "<<nw<<" subseq "<<ns<<" blocks "<<blocks<<" bits "<<bit<<'\n';id++;}std::cout<<"TOTAL "<<id<<" input_words "<<offset<<" output_bytes "<<oo<<'\n';}
