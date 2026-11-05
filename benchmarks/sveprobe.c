// SVE 指令级探针：回答 memprobe 回答不了的六个问题——
//   1. 512-bit 拷贝指令的发射上限是多少（L1 常驻源，纯指令成本，不含内存延迟）；
//   2. SVE gather（ld1d {z0.d}, p0/z, [x, z1.d]）与标量多条独立 load 谁快；
//   3. 非临时存储（stnt1d）对大块拷贝有没有收益（写侧省掉 RFO）；
//   4. 每种元素宽度一条 gather 指令能取几个元素、要搬几条 line（`lanes`）；
//   5. 4B 元素 gather 的两种下标宽度（u32 直接 / i64 收窄）分别多快（`gather ... 4 32|64`）；
//   6. 行拷贝各种行宽下 SVE 整块 vs memcpy vs 标量，预取值多少（`rowcopy`）。
//
// 单独一个文件是为了不和 memprobe.c 的口径混：这里要 `-march=armv8.2-a+sve2`，
// 同一个编译单元里开 SVE 会改掉 memprobe 其它 kernel 的自动向量化，历史数字就不可比了。
//
// 编译：
//   gcc -O3 -fopenmp -march=armv8.2-a+sve2 -msve-vector-bits=512 -o sveprobe benchmarks/sveprobe.c
// （`-msve-vector-bits=512` 与内核 setup.py 的编译选项一致；不加也能跑，但常量折叠
//   失效、lane 数变运行时值，§2.17 的历史数字本来就是加不加都一样的纯 asm 循环。）
// 用法（node 5）：
//   numactl --physcpubind=190 --membind=5 ./sveprobe copyhot 32 [sve]
//   numactl --physcpubind=190-227 --membind=5 ./sveprobe copystream 256 1|8|36 [nt]
//   numactl --physcpubind=190 --membind=5 ./sveprobe lanes
//   numactl --physcpubind=190 --membind=5 ./sveprobe gather 16K 4 32 [all|scalar|sve] 1
//   numactl --physcpubind=190 --membind=5 ./sveprobe gather 256 8 64 [all|scalar|sve] 1
//   numactl --physcpubind=190 --membind=5 ./sveprobe gather 256 4 32 [all|scalar|sve] 1
//   numactl --physcpubind=190 --membind=5 ./sveprobe gather 256 4 64 [all|scalar|sve] 1
//   numactl --physcpubind=190-227 --membind=5 ./sveprobe gather 256 4 32 all 38
//   numactl --physcpubind=190-227 --membind=5 ./sveprobe rowcopy 16 524288 8 [all|plain|memcpy|sve|svepf]
// 源大小参数：纯数字 = MB，带 K 后缀 = KB（`16K` 是 L1 常驻档）。
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <arm_sve.h>

static double now(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static uint64_t xs(uint64_t *s){uint64_t x=*s;x^=x<<13;x^=x>>7;x^=x<<17;*s=x;return x;}
static uint64_t splitmix(uint64_t *s){
    uint64_t z=(*s+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;z=(z^(z>>27))*0x94D049BB133111EBULL;return z^(z>>31);
}

// ---- SVE 基本块：一次搬 2 条 64B（128B），指针按字节推进 -------------------
// ptrue p0.d 在 512-bit VL 下等于 8 个 64-bit 车道 = 64B，正好一条 cache line。
static void sve_copy(float *dst, const float *src, size_t nbytes){
    size_t nblk=nbytes/128;
    asm volatile(
        "ptrue p0.d\n\t"
        "1:\n\t"
        "ld1d z0.d, p0/z, [%1]\n\t"
        "ld1d z1.d, p0/z, [%1, #1, mul vl]\n\t"
        "st1d z0.d, p0, [%0]\n\t"
        "st1d z1.d, p0, [%0, #1, mul vl]\n\t"
        "add %0, %0, #128\n\t"
        "add %1, %1, #128\n\t"
        "subs %2, %2, #1\n\t"
        "bne 1b\n\t"
        : "+&r"(dst), "+&r"(src), "+&r"(nblk)
        :
        : "memory");
}

// 非临时（streaming）存储：写出去不进 cache、不触发 RFO 读。
static void sve_copy_nt(float *dst, const float *src, size_t nbytes){
    size_t nblk=nbytes/128;
    asm volatile(
        "ptrue p0.d\n\t"
        "1:\n\t"
        "ld1d z0.d, p0/z, [%1]\n\t"
        "ld1d z1.d, p0/z, [%1, #1, mul vl]\n\t"
        "stnt1d z0.d, p0, [%0]\n\t"
        "stnt1d z1.d, p0, [%0, #1, mul vl]\n\t"
        "add %0, %0, #128\n\t"
        "add %1, %1, #128\n\t"
        "subs %2, %2, #1\n\t"
        "bne 1b\n\t"
        : "+&r"(dst), "+&r"(src), "+&r"(nblk)
        :
        : "memory");
}

// SVE gather 还有一条 asm 写法（`ld1d z0.d, p0/z, [x, z1.d]`），它对向量下标按
// **字节**解释；intrinsics 的 `svld1_gather_*` 按**元素**缩放。两者语义相反，
// §2.17.3 的 4.73/5.79 就是 asm 那条量出来的（当时传的是字节偏移）。下面统一走
// intrinsics，与内核一致。

// ---- 1. L1 常驻源的拷贝上限：纯指令成本（源在 32KB L1 里，重复扫） ----------
static void copyhot(size_t kb,int nt,int sve){
    size_t bytes=kb*1024;
    float *a=aligned_alloc(64,bytes),*b=aligned_alloc(64,bytes);
    for(size_t i=0;i<bytes/4;i++) a[i]=(float)i;
    size_t reps=4096;
    double t=now();
    if(sve){
        for(size_t r=0;r<reps;r++) sve_copy(b,a,bytes);
    }else{
        for(size_t r=0;r<reps;r++){
            uint64_t *d=(uint64_t*)b; const uint64_t *s=(const uint64_t*)a;
            for(size_t i=0;i<bytes/8;i++) d[i]=s[i];
        }
    }
    t=now()-t;
    double gb=2.0*bytes*reps/t/1e9;
    printf("copyhot src=%zuKB %s reps=%zu GBps=%.1f ns/64B=%.2f cyc/64B=%.1f\n",
           kb,sve?"sve512":"scalar8",reps,gb,1e9*t/(reps*bytes/64.0),
           2e9*t/(reps*bytes/64.0));
    if(b[0]!=a[0]) printf("BAD\n");
    free(a); free(b); (void)nt;
}

// ---- 2. 流式拷贝：普通 store vs 非临时 store ------------------------------
static void copystream(size_t mb,int nt,int usent){
    size_t bytes=mb<<20;
    float *a=aligned_alloc(64,bytes),*b=aligned_alloc(64,bytes);
    for(size_t i=0;i<bytes/4;i++) a[i]=(float)(i&1023);
    size_t per=bytes/(size_t)nt; per&=~(size_t)127;
    double best=1e30, t=0;      // t 必须在 parallel 外：在区内声明会变成每线程私有
    #pragma omp parallel num_threads(nt)
    {
        int tid=omp_get_thread_num();
        for(int r=0;r<3;r++){
            #pragma omp barrier
            #pragma omp single
            { t=now(); }
            #pragma omp barrier
            float *d=(float*)((char*)b+(size_t)tid*per);
            const float *s=(const float*)((char*)a+(size_t)tid*per);
            if(usent) sve_copy_nt(d,s,per); else sve_copy(d,s,per);
            #pragma omp barrier
            #pragma omp single
            { double dt=now()-t; if(dt<best) best=dt; }
        }
    }
    printf("copystream size=%zuMB threads=%d store=%s GBps=%.1f (read+write) ns/64B=%.2f\n",
           mb,nt,usent?"stnt1d":"st1d",2.0*bytes/best/1e9,1e9*best/(bytes/64.0));
    free(a); free(b);
}

// ---- 3a. 每种元素宽度一条 gather 指令取几个元素（`lanes`） ------------------
// 不查文档抄：每个宽度都真发一条 gather 并逐元素校验，校验通过的元素数就是该宽度
// 的活跃 lane 数。用 asm 而不是 intrinsic，因为 GCC 10 的 ACLE 还没暴露 ld1b/ld1h
// 的 gather（`svld1ub_gather_u32index_u32` 不存在），指令本身却在 ISA 里。
// 偏移按**字节**给（asm 形式，与 §2.17.3 一致）；binutils 2.37 要求子字形式写 uxtw。
// 必须 O0：GCC 10 下 `-O2/-O3` 会在本函数周围自动向量化，并用 z 寄存器保存活跃值，
// 而 asm 的 "z1"/"z2"/"p0" 清列没能保护它们——结果是下标被踩、随机段错误或死循环。
// 这个坑记在 RESULTS §2.23.6。
__attribute__((optimize("O0")))
static void lanes(void){
    const int N=256;
    uint8_t *src=aligned_alloc(64,N);
    for(int i=0;i<N;i++) src[i]=(uint8_t)(i*7+1);
    uint32_t boff[16], got32[16];
    uint64_t boff64[8], got64[8];
    const int esz[4]={1,2,4,8};
    int per[4]={0,0,0,0}, okk[4]={0,0,0,0};
    const int nl=(int)svcntw(), nd=(int)svcntd();

    for(int w=0;w<4;w++){
        int e=esz[w], lane_max=(e==8)?nd:nl;
        for(int i=0;i<lane_max;i++){
            uint32_t b=(uint32_t)(((i*3+1)%(N/e))*e);
            boff[i]=b; boff64[i]=b;
        }
        memset(got32,0,sizeof got32);
        if(e==1){
            asm volatile("ptrue p0.s\n\t"
                         "ld1w z2.s, p0/z, [%1]\n\t"
                         "ld1b {z1.s}, p0/z, [%0, z2.s, uxtw]\n\t"
                         "st1w z1.s, p0, [%2]\n\t"
                         :: "r"(src),"r"(boff),"r"(got32) : "memory","z1","z2","p0");
            for(int i=0;i<lane_max;i++){ if((uint32_t)src[boff[i]]==got32[i]) per[w]++; else break; }
        }else if(e==2){
            asm volatile("ptrue p0.s\n\t"
                         "ld1w z2.s, p0/z, [%1]\n\t"
                         "ld1h {z1.s}, p0/z, [%0, z2.s, uxtw]\n\t"
                         "st1w z1.s, p0, [%2]\n\t"
                         :: "r"(src),"r"(boff),"r"(got32) : "memory","z1","z2","p0");
            for(int i=0;i<lane_max;i++){
                uint16_t v; memcpy(&v,src+boff[i],2);
                if((uint32_t)v==got32[i]) per[w]++; else break;
            }
        }else if(e==4){
            asm volatile("ptrue p0.s\n\t"
                         "ld1w z2.s, p0/z, [%1]\n\t"
                         "ld1w {z1.s}, p0/z, [%0, z2.s, uxtw]\n\t"
                         "st1w z1.s, p0, [%2]\n\t"
                         :: "r"(src),"r"(boff),"r"(got32) : "memory","z1","z2","p0");
            for(int i=0;i<lane_max;i++){
                uint32_t v; memcpy(&v,src+boff[i],4);
                if(v==got32[i]) per[w]++; else break;
            }
        }else{
            memset(got64,0,sizeof got64);
            asm volatile("ptrue p0.d\n\t"
                         "ld1d z2.d, p0/z, [%1]\n\t"
                         "ld1d {z1.d}, p0/z, [%0, z2.d]\n\t"
                         "st1d z1.d, p0, [%2]\n\t"
                         :: "r"(src),"r"(boff64),"r"(got64) : "memory","z1","z2","p0");
            for(int i=0;i<lane_max;i++){
                uint64_t v; memcpy(&v,src+boff64[i],8);
                if(v==got64[i]) per[w]++; else break;
            }
        }
        okk[w]=(per[w]==lane_max);
    }

    printf("lanes VL=%d bytes svcntb=%d svcntw=%d svcntd=%d\n",
           (int)svcntb(),(int)svcntb(),(int)svcntw(),(int)svcntd());
    printf("lanes esize elems/line elems/insn lines/insn moved_B useful_B useful/moved verify\n");
    for(int w=0;w<4;w++){
        int e=esz[w], p=per[w];
        printf("lanes %5d %10d %10d %10d %7d %8d %12.4f %s\n",
               e,64/e,p,p,p*64,p*e,(double)(p*e)/(double)(p*64),okk[w]?"ok":"MISMATCH");
    }
    free(src);
}

// ---- 3b. gather：esize × 下标宽度 × 线程数 × 实现 -------------------------
// 下标语义按 ACLE：intrinsics 的 offset 是**元素**偏移（硬件按 esize 缩放），
// 与内核 svld1_gather_* 一致。文件上半部分的 asm 写法用的是**字节**偏移，两者不同。
// 两边都**写回**（8 load + 8 store），工作量对等。不带预取（内核 pf=16），
// 所以这里量的是"指令 + 访存"，不是内核耗时。
struct G { const uint8_t *src; uint8_t *dst; const int64_t *off64; const int32_t *off32; };

// 8 路展开必须**先把 8 个值读进 8 个不同寄存器再写回**。写成
// `d[i]=b[q[0]]; d[i+1]=b[q[1]]; ...` 时 GCC 会复用一个临时寄存器，把 8 条 load
// 串成一条依赖链——量出来的"标量 8 路"其实是 1 路（实测 7.6 vs 5.6 ns/元素）。
#define SC_BODY(NAME,TYPE) \
static void NAME(const struct G *g,uint64_t lo,uint64_t hi){ \
    const TYPE *b=(const TYPE*)g->src; TYPE *d=(TYPE*)g->dst; \
    const int64_t *base=g->off64; \
    for(uint64_t i=lo;i+8<=hi;i+=8){ const int64_t *q=base+i; \
        TYPE v0=b[q[0]],v1=b[q[1]],v2=b[q[2]],v3=b[q[3]], \
             v4=b[q[4]],v5=b[q[5]],v6=b[q[6]],v7=b[q[7]]; \
        d[i]=v0; d[i+1]=v1; d[i+2]=v2; d[i+3]=v3; \
        d[i+4]=v4; d[i+5]=v5; d[i+6]=v6; d[i+7]=v7; } }
SC_BODY(sc1,uint8_t) SC_BODY(sc2,uint16_t) SC_BODY(sc4,uint32_t) SC_BODY(sc8,uint64_t)

static void sv8_64(const struct G *g,uint64_t lo,uint64_t hi){
    svbool_t pg=svptrue_b64();
    for(uint64_t i=lo;i+svcntd()<=hi;i+=svcntd()){
        svuint64_t v=svld1_u64(pg,(const uint64_t*)(g->off64+i));
        svst1_u64(pg,(uint64_t*)(g->dst+i*8),svld1_gather_u64index_u64(pg,(const uint64_t*)g->src,v));
    }
}
static void sv4_32(const struct G *g,uint64_t lo,uint64_t hi){
    svbool_t pg=svptrue_b32();
    for(uint64_t i=lo;i+svcntw()<=hi;i+=svcntw()){
        svuint32_t v=svld1_u32(pg,(const uint32_t*)(g->off32+i));
        svst1_u32(pg,(uint32_t*)(g->dst+i*4),svld1_gather_u32index_u32(pg,(const uint32_t*)g->src,v));
    }
}
// 与内核 pure_gather_sve_4b_i64idx 同构：i64 下标 → uzp1 抽低 32 位 → u32index。
// 每迭代 svcntd()=8 个元素（只有一半 lane），多一条 uzp1；内核另有向量边界检查，
// 这里不做（只量上限）。
static void sv4_64(const struct G *g,uint64_t lo,uint64_t hi){
    svbool_t pg=svptrue_b64();
    svbool_t pg4=svwhilelt_b32((int64_t)0,(int64_t)svcntd());
    for(uint64_t i=lo;i+svcntd()<=hi;i+=svcntd()){
        svint64_t v=svld1_s64(pg,(const int64_t*)(g->off64+i));
        svuint32_t w=svreinterpret_u32_u64(svreinterpret_u64_s64(v));
        svst1_u32(pg4,(uint32_t*)(g->dst+i*4),
                  svld1_gather_u32index_u32(pg4,(const uint32_t*)g->src,svuzp1_u32(w,w)));
    }
}

static double run_body(int nt,int reps,void(*fn)(const struct G*,uint64_t,uint64_t),
                       const struct G *g,uint64_t n){
    double best=1e30;
    for(int r=0;r<reps;r++){
        double t0=now();
        if(nt==1){
            // 单线程直接调：套一层 `omp parallel num_threads(1)` 会掉 40%
            // （body 被 outlined 成 *_omp_fn，跨循环的调度优化没了）。§2.17 的
            // 单线程数字也是直接循环，这样才可比。
            fn(g,0,n);
        }else{
            #pragma omp parallel num_threads(nt)
            {
                // 对齐 64 个元素：8B 走 8 lane、4B 走 16 lane，都能整除
                uint64_t per=(n/(uint64_t)nt)&~(uint64_t)63;
                uint64_t lo=(uint64_t)omp_get_thread_num()*per;
                uint64_t hi=(omp_get_thread_num()==nt-1)?n:lo+per;
                fn(g,lo,hi);
            }
        }
        double dt=now()-t0;
        if(dt<best) best=dt;
    }
    return best;
}

static void gathertest(size_t sz,int esize,int idxbits,int nt,int doscalar,int dosve,int kb){
    size_t nbytes=kb?((size_t)sz<<10):((size_t)sz<<20);
    size_t n=nbytes/(size_t)esize;
    uint8_t *src=aligned_alloc(64,nbytes);
    for(size_t i=0;i<nbytes;i++) src[i]=(uint8_t)((i*131+7)&0xff);
    uint64_t idxn=1u<<20;                       // 下标表 1M 项
    int64_t *off64=malloc(idxn*sizeof(int64_t));
    int32_t *off32=malloc(idxn*sizeof(int32_t));
    uint64_t g=0x51ED270B1A7C3D5FULL+(uint64_t)nbytes*131+(uint64_t)esize*17+(uint64_t)idxbits;
    for(uint64_t i=0;i<idxn;i++){
        int64_t v=(int64_t)(splitmix(&g)%n);
        off64[i]=v; off32[i]=(int32_t)v;
    }
    uint8_t *dst=aligned_alloc(64,(size_t)idxn*(size_t)esize+64);
    struct G ctx={src,dst,off64,off32};
    const int R=4;
    double t;
    const char *szl=kb?"KB":"MB";

    if(doscalar){
        void(*fn)(const struct G*,uint64_t,uint64_t)= esize==1?sc1:esize==2?sc2:esize==4?sc4:sc8;
        memset(dst,0,(size_t)idxn*(size_t)esize);
        t=run_body(nt,R,fn,&ctx,idxn);
        int chk=1;
        for(uint64_t i=0;i<idxn;i++)
            if(memcmp(dst+i*(size_t)esize,src+(size_t)off64[i]*(size_t)esize,(size_t)esize)!=0){chk=0;break;}
        printf("gather sv=%zu%s esize=%d idx=%dbit threads=%d impl=scalar chains=8 ns/access=%.2f cyc/access=%.1f correct=%d\n",
               sz,szl,esize,idxbits,nt,1e9*t/(double)idxn,2e9*t/(double)idxn,chk);
    }

    if(dosve){
        int lanes;
        void(*fn)(const struct G*,uint64_t,uint64_t);
        if(esize==8){ fn=sv8_64; lanes=(int)svcntd(); }
        else if(esize==4 && idxbits==32){ fn=sv4_32; lanes=(int)svcntw(); }
        else if(esize==4 && idxbits==64){ fn=sv4_64; lanes=(int)svcntd(); }
        else { printf("gather esize=%d idx=%dbit impl=sve unsupported\n",esize,idxbits);
               free(src); free(dst); free(off64); free(off32); return; }
        memset(dst,0,(size_t)idxn*(size_t)esize);
        t=run_body(nt,R,fn,&ctx,idxn);
        int chk=1;
        for(uint64_t i=0;i<idxn;i++)
            if(memcmp(dst+i*(size_t)esize,src+(size_t)off64[i]*(size_t)esize,(size_t)esize)!=0){chk=0;break;}
        printf("gather sv=%zu%s esize=%d idx=%dbit threads=%d impl=sve512 lanes=%d ns/access=%.2f cyc/access=%.1f correct=%d\n",
               sz,szl,esize,idxbits,nt,lanes,1e9*t/(double)idxn,2e9*t/(double)idxn,chk);
    }
    free(src); free(dst); free(off64); free(off32);
}

// ---- 3c. 行拷贝：SVE 整块 vs memcpy vs 标量，带/不带预取 --------------------
// 随机行序、源 16MB、输出 rows 行，与 rowcopy_sve 的访存形态同构（不含 linearize）。
static inline void copy_sve_bytes(uint8_t *d,const uint8_t *s,size_t n){
    size_t i=0;
    for(;i+svcntb()<=n;i+=svcntb()){
        svbool_t p=svptrue_b8();
        svst1_u8(p,d+i,svld1_u8(p,s+i));
    }
    if(i<n){
        svbool_t p=svwhilelt_b8((uint64_t)i,(uint64_t)n);
        svst1_u8(p,d+i,svld1_u8(p,s+i));
    }
}
static inline void copy_plain_bytes(uint8_t *d,const uint8_t *s,size_t n){
    size_t i=0;
    for(;i+8<=n;i+=8){ uint64_t v; memcpy(&v,s+i,8); memcpy(d+i,&v,8); }
    for(;i<n;i++) d[i]=s[i];
}
static void rowcopy(int a,long rows,int nt,int mode){
    // mode: 0=all 1=plain 2=memcpy 3=sve 4=svepf
    size_t srcbytes=16u<<20;
    size_t srows=srcbytes/(size_t)a;
    uint8_t *src=aligned_alloc(64,srcbytes),*dst=aligned_alloc(64,(size_t)rows*(size_t)a);
    for(size_t i=0;i<srcbytes;i++) src[i]=(uint8_t)((i*17+3)&0xff);
    int32_t *ridx=malloc((size_t)rows*sizeof(int32_t));
    uint64_t g=0x9E3779B97F4A7C15ULL+(uint64_t)a*7919+(uint64_t)rows;
    for(long i=0;i<rows;i++) ridx[i]=(int32_t)(splitmix(&g)%srows);

    const char *names[5]={"plain","memcpy","sve","svepf","?"};
    const int pfdist=8;
    double best[5]; for(int m=0;m<5;m++) best[m]=1e30;

    for(int rep=0;rep<3;rep++){
        for(int m=1;m<=4;m++){
            if(mode!=0 && mode!=m) continue;
            double t=now();
            #pragma omp parallel num_threads(nt)
            {
                #pragma omp for schedule(static)
                for(long i=0;i<rows;i++){
                    const uint8_t *s=src+(size_t)ridx[i]*a;
                    uint8_t *d=dst+(size_t)i*a;
                    if(m==3 || m==4){
                        if(m==4 && i+pfdist<rows)
                            __builtin_prefetch(src+(size_t)ridx[i+pfdist]*a,0,3);
                        copy_sve_bytes(d,s,a);
                    }else if(m==2){
                        memcpy(d,s,a);
                    }else{
                        copy_plain_bytes(d,s,a);
                    }
                }
            }
            t=now()-t;
            if(t<best[m]) best[m]=t;
        }
    }
    int chk=1;
    for(long i=0;i<rows && chk;i++)
        if(memcmp(dst+(size_t)i*a,src+(size_t)ridx[i]*a,a)!=0) chk=0;
    for(int m=1;m<=4;m++){
        if(mode!=0 && mode!=m) continue;
        printf("rowcopy A=%d rows=%ld threads=%d impl=%s ns/row=%.2f GBps_useful=%.1f correct=%d\n",
               a,rows,nt,names[m-1],1e9*best[m]/rows,(double)rows*(double)a/best[m]/1e9,chk);
    }
    free(src); free(dst); free(ridx);
}

int main(int argc,char**argv){
    if(argc<2){
        fprintf(stderr,"usage: %s copyhot KB [sve] | copystream MB THREADS [nt] | lanes | "
                       "gather SIZE[K] ESIZE(1|2|4|8) IDXBITS(32|64) [all|scalar|sve] [THREADS] | "
                       "rowcopy A ROWS THREADS [all|plain|memcpy|sve|svepf]\n",argv[0]);
        return 1;
    }
    if(!strcmp(argv[1],"copyhot")) copyhot((size_t)atol(argv[2]),1,argc>3&&!strcmp(argv[3],"sve"));
    else if(!strcmp(argv[1],"copystream")) copystream((size_t)atol(argv[2]),atoi(argv[3]),argc>4&&!strcmp(argv[4],"nt"));
    else if(!strcmp(argv[1],"lanes")) lanes();
    else if(!strcmp(argv[1],"gather")){
        int esize=argc>3?atoi(argv[3]):8;
        int idxbits=argc>4?atoi(argv[4]):(esize==8?64:32);
        const char *w=argc>5?argv[5]:"all";
        // 源大小：纯数字 = MB；带 K 后缀 = KB（16K 即 L1 常驻，用来和 §2.20 对齐）
        char *end=0;
        long v=strtol(argv[2],&end,10);
        size_t mb=(size_t)v;
        int kb_mode=0;
        if(end && (*end=='k'||*end=='K')){ kb_mode=1; }
        int nt=argc>6?atoi(argv[6]):1;
        gathertest(mb,esize,idxbits,nt,
                   !strcmp(w,"all")||!strcmp(w,"scalar"),
                   !strcmp(w,"all")||!strcmp(w,"sve"),kb_mode);
    }
    else if(!strcmp(argv[1],"rowcopy")){
        int a=argc>2?atoi(argv[2]):16;
        long rows=argc>3?atol(argv[3]):200000;
        int nt=argc>4?atoi(argv[4]):1;
        const char *w=argc>5?argv[5]:"all";
        rowcopy(a,rows,nt,!strcmp(w,"plain")?1:!strcmp(w,"memcpy")?2:!strcmp(w,"sve")?3:
                       !strcmp(w,"svepf")?4:0);
    }
    else { fprintf(stderr,"unknown subcommand %s\n",argv[1]); return 1; }
    return 0;
}
