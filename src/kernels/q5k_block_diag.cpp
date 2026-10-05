// Per-block Q5_K x Q8_K scalar-min contract diagnostic for a version-2 stage
// fixture with the selected expert blob embedded in the payload.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

struct H {
    uint64_t magic;
    uint32_t version, header_bytes;
    uint64_t payload_bytes, payload_fnv1a;
    int64_t pos0, layer, expert, route_index;
    float route_weight;
    uint64_t n_embd, n_ff, k, act_bytes, h_bytes, blob_bytes;
    uint32_t act_type, gu_type, d_type, gu_act, d_act, reserved;
};
static uint64_t fnv(const std::vector<uint8_t>& p) { uint64_t h=1469598103934665603ull; for(uint8_t x:p){h^=x;h*=1099511628211ull;} return h; }
static uint32_t bits(float x) { uint32_t u; std::memcpy(&u,&x,4); return u; }
static float f16(uint16_t h) {
    const uint32_t s=(h>>15)&1, e=(h>>10)&31, m=h&1023;
    uint32_t u;
    if (!e) return s ? -std::ldexp((float)m, -24) : std::ldexp((float)m, -24);
    else if (e == 31) u = (s<<31) | 0x7f800000u | (m<<13);
    else u = (s<<31) | ((e + 112) << 23) | (m<<13);
    float x; std::memcpy(&x,&u,4); return x;
}
static void unpack(const uint8_t* src, uint8_t sc[8], uint8_t mn[8]) {
    constexpr uint32_t km1=0x3f3f3f3f, km2=0x0f0f0f0f, km3=0x03030303;
    uint32_t u[4]{}; std::memcpy(u,src,12);
    u[3]=((u[2]>>4)&km2)|(((u[1]>>6)&km3)<<4);
    const uint32_t aux=u[1]&km1;
    u[1]=(u[2]&km2)|(((u[0]>>6)&km3)<<4); u[2]=aux; u[0]&=km1;
    std::memcpy(sc,u,8); std::memcpy(mn,u+2,8);
}
static int32_t iadd(int32_t a,int32_t b){return a+b;}
int main(int argc,char**argv){
    if(argc<2){std::cerr<<"usage: q5k_block_diag FIXTURE_V2 [ROW...]\n";return 2;}
    std::ifstream in(argv[1],std::ios::binary); H h{}; in.read((char*)&h,sizeof h);
    if(!in||h.magic!=0x3147545341525453ull||h.version!=2){std::cerr<<"bad v2 fixture\n";return 2;}
    std::vector<uint8_t> p((size_t)h.payload_bytes); in.read((char*)p.data(),p.size());
    if(!in||fnv(p)!=h.payload_fnv1a){std::cerr<<"fixture checksum failed\n";return 2;}
    const size_t ids=h.k*4, weights=h.k*4, base=ids+weights+h.act_bytes;
    const size_t n=h.n_ff, act_off=ids+weights, g_off=base, gg_off=g_off+n*4, u_off=gg_off+n*4, gu_off=u_off+n*4;
    const size_t h_off=gu_off+n*4, gh_off=h_off+n*4, qcpu_off=gh_off+n*4, qgpu_off=qcpu_off+h.h_bytes;
    const size_t out_off=qgpu_off+h.h_bytes, gout_off=out_off+h.n_embd*4, cd_off=gout_off+h.n_embd*4;
    const size_t gd_off=cd_off+h.n_embd*4, ggd_off=gd_off+h.n_embd*4, blob_off=ggd_off+h.n_embd*4;
    if(blob_off+h.blob_bytes>p.size()){std::cerr<<"fixture layout failed\n";return 2;}
    const auto* act=p.data()+act_off; const auto* blob=p.data()+blob_off;
    const size_t q5sz=sizeof(uint16_t)*2+12+128+32, gu_row=h.n_embd/256*q5sz, up_off=h.n_ff*gu_row;
    bool up=false; int arg0=2; if(arg0<argc && std::string(argv[arg0])=="up"){up=true; ++arg0;}
    std::vector<int> rows; for(int i=arg0;i<argc;++i) rows.push_back(std::stoi(argv[i])); if(rows.empty()) rows={2};
    const uint8_t* matrix=blob+(up?h.n_ff*gu_row:0); const size_t cpu_off=up?u_off:g_off, gpu_off=up?gu_off:gg_off;
    std::cout<<"fixture_version="<<h.version<<" layer="<<h.layer<<" expert="<<h.expert<<" matrix="<<(up?"up":"gate")<<" rows="<<rows.size()<<" q5_block_bytes="<<q5sz<<" gu_row="<<gu_row<<" blocks="<<(h.n_embd/256)<<"\n";
    for(int row:rows){
        const uint8_t* wrow=matrix+(size_t)row*gu_row;
        const float* cpu=(const float*)(p.data()+cpu_off+(size_t)row*4);
        const float* gpu=(const float*)(p.data()+gpu_off+(size_t)row*4);
        std::cout<<"row="<<row<<" cpu_gate_bits=0x"<<std::hex<<bits(*cpu)<<" gpu_gate_bits=0x"<<bits(*gpu)<<std::dec<<"\n";
        float main[8]{}, oldm[4]{}, summs=0;
        float cpu_sums[8]{}, cpu_mins=0;
        for(size_t kbx=0;kbx<h.n_embd/256;++kbx){
            const uint8_t* wb=wrow+kbx*q5sz; const uint8_t* ab=act+kbx*292;
            const uint16_t wd=(uint16_t)wb[0]|((uint16_t)wb[1]<<8), wm=(uint16_t)wb[2]|((uint16_t)wb[3]<<8);
            const float d=f16(wd)*((const float*)ab)[0]; // block_q8_K.d is the first float
            const float dmin=-f16(wm)*((const float*)ab)[0];
            const uint8_t* scales=wb+4; const uint8_t* qh=wb+16; const uint8_t* qs=wb+48;
            uint8_t sc[8],mn[8]; unpack(scales,sc,mn);
            const int8_t* q8=(const int8_t*)(ab+4); int si[8]{},mi[4]{};
            int8_t aux[256];
            for(int j=0;j<4;++j){
                for(int l=0;l<32;++l) aux[j*64+l]=(int8_t)((qs[j*32+l]&15)|(((qh[l]>>(2*j))&1)<<4));
                for(int l=0;l<32;++l) aux[j*64+32+l]=(int8_t)((qs[j*32+l]>>4)|(((qh[l]>>(2*j+1))&1)<<4));
            }
            int32_t aux32[8]{};
            for(int j=0;j<8;++j){ for(int l=0;l<8;++l) for(int t=0;t<4;++t) aux32[l] += (int32_t)sc[j]*(int32_t)q8[j*32+t*8+l]*(int32_t)aux[j*32+t*8+l]; }
            for(int l=0;l<8;++l) cpu_sums[l]=fmaf(d,(float)aux32[l],cpu_sums[l]);
            int32_t sumi=0; for(int j=0;j<16;++j) sumi += ((const int16_t*)(ab+260))[j]*(int)mn[j/2];
            cpu_mins -= f16(wm)*((const float*)ab)[0]*(float)sumi;
            for(int lane=0;lane<8;++lane){ for(int j=0;j<4;++j){int lo=0,hi=0;for(int q=0;q<4;++q){int qi=4*lane+q;int ql=qs[j*32+qi]&15,qh5=qs[j*32+qi]>>4; ql|=((qh[qi]>>(2*j))&1)<<4; qh5|=((qh[qi]>>(2*j+1))&1)<<4; lo+=ql*q8[j*64+qi]; hi+=qh5*q8[j*64+32+qi];} si[lane]+=sc[2*j]*lo+sc[2*j+1]*hi;} main[lane]=fmaf(d,(float)si[lane],main[lane]);}
            for(int lane=0;lane<4;++lane){int g=lane*4;int q0=(int)((const int16_t*)(ab+260))[g]+(int)((const int16_t*)(ab+260))[g+1];int q1=(int)((const int16_t*)(ab+260))[g+2]+(int)((const int16_t*)(ab+260))[g+3];mi[lane]=mn[2*lane]*q0+mn[2*lane+1]*q1;oldm[lane]=fmaf(dmin,(float)mi[lane],oldm[lane]);}
            const int total=iadd(iadd(mi[0],mi[1]),iadd(mi[2],mi[3])); summs=fmaf(dmin,(float)total,summs);
            std::cout<<"block="<<kbx<<" d_bits=0x"<<std::hex<<bits(d)<<" dmin_bits=0x"<<bits(dmin)<<" mi="<<mi[0]<<","<<mi[1]<<","<<mi[2]<<","<<mi[3]<<" total="<<std::dec<<total<<" summs_bits=0x"<<std::hex<<bits(summs)<<" oldm_bits=0x"<<bits(oldm[0])<<","<<bits(oldm[1])<<","<<bits(oldm[2])<<","<<bits(oldm[3])<<std::dec<<"\n";
        }
        float r0=main[0]+main[4], r1=main[1]+main[5], r2=main[2]+main[6], r3=main[3]+main[7];
        float main_h=(r0+r2)+(r1+r3);
        float cpu_main=0; for(int l=0;l<8;++l) cpu_main += cpu_sums[l];
        std::cout<<"row="<<row<<" main_bits=0x"<<std::hex<<bits(main_h)<<" summs_bits=0x"<<bits(summs)<<" fixed_result_bits=0x"<<bits(main_h+summs)<<" old_mins_bits=0x"<<bits(oldm[0]+oldm[2]+oldm[1]+oldm[3])<<" cpu_main_bits=0x"<<bits(cpu_main)<<" cpu_mins_bits=0x"<<bits(cpu_mins)<<" cpu_recon_bits=0x"<<bits(cpu_main+cpu_mins)<<" cpu_bits=0x"<<bits(*cpu)<<" gpu_bits=0x"<<bits(*gpu)<<std::dec<<"\n";
    }
}
