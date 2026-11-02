// SVE 指令级探针：回答 memprobe 回答不了的三个问题——
//   1. 512-bit 拷贝指令的发射上限是多少（L1 常驻源，纯指令成本，不含内存延迟）；
//   2. SVE gather（ld1d {z0.d}, p0/z, [x, z1.d]）与标量多条独立 load 谁快；
//   3. 非临时存储（stnt1d）对大块拷贝有没有收益（写侧省掉 RFO）。
//
// 单独一个文件是为了不和 memprobe.c 的口径混：这里要 `-march=armv8.2-a+sve2`，
// 同一个编译单元里开 SVE 会改掉 memprobe 其它 kernel 的自动向量化，历史数字就不可比了。
//
// 编译：
//   gcc -O3 -fopenmp -march=armv8.2-a+sve2 -o sveprobe benchmarks/sveprobe.c
// 用法（node 5）：
//   numactl --physcpubind=190 --membind=5 ./sveprobe copyhot 32 [2]
//   numactl --physcpubind=190-227 --membind=5 ./sveprobe copystream 256 1|8|36 [nt]
//   numactl --physcpubind=190 --membind=5 ./sveprobe gather 256 20000000
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

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

// SVE gather：每轮一条 line（VL=64B）。实测这套工具链/指令语义把向量下标按
// **字节**解释（不是按元素大小缩放），所以调用方传字节偏移——这一点是量出来的，
// 见 RESULTS §2.17.3。
static void sve_gather_d(double *dst, const void *base, const int64_t *off, size_t n){
    asm volatile(
        "ptrue p0.d\n\t"
        "1:\n\t"
        "ld1d z1.d, p0/z, [%2]\n\t"
        "ld1d z0.d, p0/z, [%1, z1.d]\n\t"
        "st1d z0.d, p0, [%0]\n\t"
        "add %0, %0, #64\n\t"
        "add %2, %2, #64\n\t"
        "subs %3, %3, #1\n\t"
        "bne 1b\n\t"
        : "+&r"(dst), "+&r"(base), "+&r"(off), "+&r"(n)
        :
        : "memory");
}

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

// ---- 3. SVE gather vs 标量 gather（同源、同下标、同工作集） ----------------
// 两边都扫完整张下标表（idxn 项）为一轮，重复 R 轮取总时间，避免一次性冷启动。
static void gathertest(size_t mb,int esize){
    size_t nbytes=mb<<20;
    size_t n=nbytes/8;                     // 8B 元素
    double *src=aligned_alloc(64,nbytes);
    for(size_t i=0;i<n;i++) src[i]=(double)(i&1023);
    uint64_t idxn=1<<20;                    // 下标表 1M 项（8MB），顺序扫完
    int64_t *offb=malloc(idxn*sizeof(int64_t));   // 字节偏移
    int64_t *offe=malloc(idxn*sizeof(int64_t));   // 元素偏移（校验用）
    uint64_t g=0x51ED270B1A7C3D5FULL+mb+esize;
    for(uint64_t i=0;i<idxn;i++){
        int64_t v=(int64_t)(splitmix(&g)%n);
        offe[i]=v; offb[i]=v*8;
    }
    double *dst=aligned_alloc(64,(size_t)idxn*sizeof(double)+64);
    int R=8;
    double t; double sink=0;

    // 标量：chains=8 条独立 load。每轮换起点，防止编译器把 8 轮 CSE 成一轮。
    t=now();
    for(int r=0;r<R;r++){
        uint64_t base=r*(idxn/8);
        for(uint64_t k=0;k<idxn;k+=8){
            const int64_t *q=offe+((k+base)&(idxn-8));
            sink+=src[q[0]]+src[q[1]]+src[q[2]]+src[q[3]]
                 +src[q[4]]+src[q[5]]+src[q[6]]+src[q[7]];
        }
    }
    t=now()-t;
    double nacc=(double)R*idxn;
    printf("gather sv=%zuMB esize=8 impl=scalar chains=8 ns/access=%.2f cyc/access=%.1f sink=%g\n",
           mb,1e9*t/nacc,2e9*t/nacc,sink);

    t=now();
    for(int r=0;r<R;r++) sve_gather_d(dst,src,offb,idxn/8);
    t=now()-t;
    int chk=1;
    for(uint64_t i=0;i<idxn;i++) if(dst[i]!=src[offe[i]]){chk=0;break;}
    printf("gather sv=%zuMB esize=8 impl=sve512 chains=8 ns/access=%.2f cyc/access=%.1f correct=%d\n",
           mb,1e9*t/nacc,2e9*t/nacc,chk);
    free(src); free(dst); free(offb); free(offe);
}

int main(int argc,char**argv){
    if(argc<2){
        fprintf(stderr,"usage: %s copyhot KB [sve] | copystream MB THREADS nt | gather MB [8]\n",argv[0]);
        return 1;
    }
    if(!strcmp(argv[1],"copyhot")) copyhot((size_t)atol(argv[2]),1,argc>3&&!strcmp(argv[3],"sve"));
    else if(!strcmp(argv[1],"copystream")) copystream((size_t)atol(argv[2]),atoi(argv[3]),argc>4&&!strcmp(argv[4],"nt"));
    else if(!strcmp(argv[1],"gather")) gathertest((size_t)atol(argv[2]),argc>3?atoi(argv[3]):8);
    return 0;
}
