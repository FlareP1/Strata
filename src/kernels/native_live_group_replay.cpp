// Debug-only replay of the exact live layer-15 pcie group.  This keeps the four distinct expert blobs in their
// router order and uses the captured Q8_K activation, destination rows and one-token group plan.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
namespace cpu = strata::kernels::cpu;
struct DH { uint64_t magic; uint32_t version,T; int64_t pos0,layer; uint64_t N,FF,K,act_bytes; uint32_t act_type,meta_words; };
static void ok(bool x,const char* s){if(!x){std::fprintf(stderr,"%s\n",s);std::exit(2);}}
static void cu(cudaError_t e,const char* s){if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",s,cudaGetErrorString(e));std::exit(2);}}
int main(int argc,char** argv){
 if(argc!=6){std::fprintf(stderr,"usage: %s MODEL DETAILS POS0 LAYER route0,route1,...\n",argv[0]);return 2;}
 const std::string mp=argv[1],dp=argv[2]; const int64_t pos=std::stoll(argv[3]),layer=std::stoll(argv[4]);
 std::vector<int> routes; for(size_t p=0;p<std::strlen(argv[5]);){size_t q=p;while(q<std::strlen(argv[5])&&argv[5][q]!=',')++q;routes.push_back(std::stoi(std::string(argv[5]+p,q-p)));p=q+1;}
 std::ifstream in(dp,std::ios::binary);ok((bool)in,"details open");DH h{};std::vector<float>w;std::vector<int32_t>ids,meta;std::vector<uint8_t>act;bool found=false;
 while(in.read((char*)&h,sizeof(h))){ok(h.magic==0x535452414d4f4532ull,"details magic");w.resize(h.K);ids.resize(h.K);act.resize(h.act_bytes);meta.resize(h.meta_words);in.read((char*)w.data(),w.size()*4);in.read((char*)ids.data(),ids.size()*4);in.read((char*)act.data(),act.size());in.read((char*)meta.data(),meta.size()*4);ok((bool)in,"details truncated");if(h.pos0==pos&&h.layer==layer){found=true;break;}}
 ok(found,"target record missing");ok(h.act_type==15,"requires Q8_K activation");ok(h.N==2560&&h.FF==640,"unexpected geometry");
 const strata::GgufModel model(strata::gguf_split_paths(mp));const strata::TensorInfo*t[3]={};const uint8_t*d[3]={};const char*roles[3]={"gate","up","down"};
 for(int r=0;r<3;++r){size_t at=0;t[r]=model.find("blk."+std::to_string(layer)+".ffn_"+roles[r]+"_exps.weight",&at);if(t[r])d[r]=model.shard(at).tensor_data(*t[r]);}
 ok(t[0]&&t[1]&&t[2],"expert tensors missing");cpu::NativeFmt f;std::string err;ok(cpu::native_fmt((int)t[0]->type,(int)t[2]->type,h.N,h.FF,f,err),err.c_str());ok(f.gu_act==15&&f.d_type==7&&f.d_act==9,"not Q4_K/Q5_1");
 std::vector<std::vector<uint8_t>>blobs(routes.size());std::vector<std::vector<float>>cpuout(routes.size(),std::vector<float>(h.N));std::vector<std::vector<uint8_t>>hq(routes.size(),std::vector<uint8_t>(f.h_bytes));
 for(size_t j=0;j<routes.size();++j){int e=ids[routes[j]];ok(e>=0,"bad expert");blobs[j].resize(f.bytes);std::memcpy(blobs[j].data(),d[0]+(size_t)e*f.up_off,f.up_off);std::memcpy(blobs[j].data()+f.up_off,d[1]+(size_t)e*f.up_off,f.up_off);std::memcpy(blobs[j].data()+f.down_off,d[2]+(size_t)e*(f.bytes-f.down_off),f.bytes-f.down_off);std::vector<float>hh(f.n_ff);const void*ap[1]={act.data()};float*hp[1]={hh.data()};const void*hqp[1]={hq[j].data()};float*op[1]={cpuout[j].data()};cpu::native_gu_rows(f,blobs[j].data(),ap,1,hp,0,f.n_ff);cpu::native_quant_h(f,hh.data(),hq[j].data());cpu::native_down_rows(f,blobs[j].data(),hqp,1,op,0,f.n_embd);}
 cudaStream_t s=nullptr;cu(cudaStreamCreate(&s),"stream");std::vector<void*>db(routes.size());void* da=nullptr;void*ds=nullptr;float*do_=nullptr;unsigned long long*dpv=nullptr;int32_t*dst=nullptr;int32_t*dn=nullptr;int32_t*dd=nullptr;int32_t*dt=nullptr;
 for(size_t j=0;j<routes.size();++j){cu(cudaMalloc(&db[j],blobs[j].size()),"blob alloc");cu(cudaMemcpy(db[j],blobs[j].data(),blobs[j].size(),cudaMemcpyHostToDevice),"blob copy");}
 cu(cudaMalloc(&da,h.act_bytes),"act alloc");cu(cudaMemcpy(da,act.data(),h.act_bytes,cudaMemcpyHostToDevice),"act copy");cu(cudaMalloc(&ds,strata::kernels::native_expert_scratch_bytes(routes.size(),f.n_ff)),"scratch");cu(cudaMalloc((void**)&do_,h.K*h.N*4),"out");cu(cudaMalloc((void**)&dpv,routes.size()*8),"ptrs");cu(cudaMalloc((void**)&dst,(routes.size()+1)*4),"starts");cu(cudaMalloc((void**)&dn,4),"groups");cu(cudaMalloc((void**)&dd,routes.size()*4),"dst");cu(cudaMalloc((void**)&dt,routes.size()*4),"tok");
 std::vector<unsigned long long>pv(routes.size());for(size_t j=0;j<routes.size();++j)pv[j]=(unsigned long long)db[j];std::vector<int32_t>st(routes.size()+1),di(routes.size()),ti(routes.size());for(size_t j=0;j<routes.size();++j){st[j]=j;di[j]=routes[j];ti[j]=0;}st.back()=routes.size();int ng=routes.size();cu(cudaMemcpy(dpv,pv.data(),pv.size()*8,cudaMemcpyHostToDevice),"ptrs copy");cu(cudaMemcpy(dst,st.data(),st.size()*4,cudaMemcpyHostToDevice),"starts copy");cu(cudaMemcpy(dn,&ng,4,cudaMemcpyHostToDevice),"groups copy");cu(cudaMemcpy(dd,di.data(),di.size()*4,cudaMemcpyHostToDevice),"dst copy");cu(cudaMemcpy(dt,ti.data(),ti.size()*4,cudaMemcpyHostToDevice),"tok copy");
 auto L=strata::kernels::native_expert_layout(f.gu_type,f.d_type,f.n_embd,f.n_ff,f.gu_act,f.d_act);strata::kernels::native_expert_grouped(L,dpv,dst,dn,dd,dt,ng,routes.size(),da,h.act_bytes,ds,do_,s);cu(cudaStreamSynchronize(s),"sync");std::vector<float>go(h.K*h.N);cu(cudaMemcpy(go.data(),do_,go.size()*4,cudaMemcpyDeviceToHost),"output copy");
 for(size_t j=0;j<routes.size();++j){int r=routes[j];double mx=0,s2=0;for(size_t i=0;i<h.N;++i){double d0=go[(size_t)r*h.N+i]-cpuout[j][i];mx=std::max(mx,std::fabs(d0));s2+=d0*d0;}std::cout<<"route="<<r<<" expert="<<ids[r]<<" weight="<<w[r]<<" max="<<mx<<" rms="<<std::sqrt(s2/h.N)<<" gpu_first=";for(int i=0;i<8;++i)std::cout<<(i?",":"")<<go[(size_t)r*h.N+i];std::cout<<"\n";}
 for(void*p:db)cudaFree(p);cudaFree(da);cudaFree(ds);cudaFree(do_);cudaFree(dpv);cudaFree(dst);cudaFree(dn);cudaFree(dd);cudaFree(dt);cudaStreamDestroy(s);return 0;
}
