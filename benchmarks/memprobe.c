#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <sys/mman.h>

static double now(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static uint64_t xs(uint64_t *s){uint64_t x=*s;x^=x<<13;x^=x>>7;x^=x<<17;*s=x;return x;}

static void latency(size_t size, uint64_t acc){
    uint64_t n=size/8; uint64_t *p=aligned_alloc(64,size);
    for(uint64_t i=0;i<n;i++) p[i]=i;
    uint64_t s=88172645463325252ULL;
    for(uint64_t i=n-1;i>0;i--){uint64_t j=xs(&s)%(i+1);uint64_t t=p[i];p[i]=p[j];p[j]=t;}
    volatile uint64_t idx=0;
    for(uint64_t i=0;i<acc/10+10;i++) idx=p[idx];
    double t0=now();
    for(uint64_t i=0;i<acc;i++) idx=p[idx];
    double t=now()-t0;
    printf("latency size=%zuKB accesses=%lu ns/access=%.2f implied_MBps=%.1f sink=%lu\n",
           size/1024,acc,1e9*t/acc,(double)acc*64/t/1e6,(unsigned long)idx);
    free(p);
}

// 单遍计时会把 OpenMP 建队 + TLB 冷启动算进去：256MB/8t 只有 ~3ms，误差可达 30%。
// REPS>1 时在同一个 parallel 区内重复扫描，取最好的一遍。
static void bandwidth(size_t size,int nt,int wr,int reps){
    size_t n=size/8; uint64_t *a=aligned_alloc(64,size); uint64_t *b=wr?aligned_alloc(64,size):NULL;
    for(size_t i=0;i<n;i++) a[i]=i*3+1;
    if(wr) memset(b,0,size);
    double t=0, best=0; double bytes=(double)size*(wr?2:1);
    if(reps<1) reps=1;
    #pragma omp parallel num_threads(nt)
    {
        size_t lo=(size_t)((double)n*omp_get_thread_num()/nt), hi=(size_t)((double)n*(omp_get_thread_num()+1)/nt);
        for(int r=0;r<reps;r++){
            #pragma omp barrier
            #pragma omp single
            { t=now(); }
            #pragma omp barrier
            if(!wr){uint64_t s0=0,s1=0,s2=0,s3=0;
                for(size_t i=lo;i+3<hi;i+=4){s0+=a[i];s1+=a[i+1];s2+=a[i+2];s3+=a[i+3];}
                if(omp_get_thread_num()==0) a[0]+=s0+s1+s2+s3;
            }else{
                for(size_t i=lo;i<hi;i++) b[i]=a[i]*3+1;
                if(omp_get_thread_num()==0) b[0]+=1;
            }
            #pragma omp barrier
            #pragma omp single
            {
                double gb=bytes/(now()-t)/1e9;
                if(gb>best) best=gb;
                if(reps>1) printf("bandwidth size=%zuMB threads=%d op=%s rep=%d GBps=%.2f\n",
                                  size/1024/1024,nt,wr?"copy":"read",r,gb);
            }
        }
    }
    printf("bandwidth size=%zuMB threads=%d op=%s reps=%d GBps=%.2f\n",
           size/1024/1024,nt,wr?"copy":"read",reps,best);
    if(wr && b[0]==0xdeadbeef) printf("x");
    free(a); free(b);
}

static uint64_t splitmix(uint64_t *s){
    uint64_t z=(*s+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;z=(z^(z>>27))*0x94D049BB133111EBULL;return z^(z>>31);
}

static void gather(size_t size,uint64_t acc,int mode){
    size_t n=size/4; float *src=aligned_alloc(64,size);
    for(size_t i=0;i<n;i++) src[i]=(float)(i&1023);
    // 种子必须用 splitmix 拉开：直接取 1..8 会让 8 路 xorshift 序列相关，
    // 8 条"独立流"落到同样的行上，测出来的随机 gather 偏快（曾差 30%+）。
    uint64_t g=0x243F6A8885A308D3ULL+size+mode, s0=splitmix(&g),s1=splitmix(&g),s2=splitmix(&g),s3=splitmix(&g),
             s4=splitmix(&g),s5=splitmix(&g),s6=splitmix(&g),s7=splitmix(&g);
    uint64_t m=n-1;
    volatile float sink=0; double t0=0;
    if(mode==0){ // sequential
        t0=now();
        for(uint64_t i=0;i<acc;i++) sink+=src[i&m];
    } else if(mode==1){ // one element per 64B line, sequential lines
        uint64_t step=(n>16?16:1);
        t0=now();
        for(uint64_t i=0;i<acc;i++) sink+=src[(i*step)&m];
    } else { // random, 8 independent streams (MLP)
        t0=now();
        for(uint64_t i=0;i+7<acc;i+=8){
            sink+=src[xs(&s0)&m]; sink+=src[xs(&s1)&m]; sink+=src[xs(&s2)&m]; sink+=src[xs(&s3)&m];
            sink+=src[xs(&s4)&m]; sink+=src[xs(&s5)&m]; sink+=src[xs(&s6)&m]; sink+=src[xs(&s7)&m];
        }
    }
    double t=now()-t0; const char*names[]={"seq","stride16","random"};
    printf("gather size=%zuMB accesses=%lu mode=%s ns/access=%.2f useful_GBps=%.2f line_GBps=%.2f sink=%f\n",
        size/1024/1024,acc,names[mode],1e9*t/acc,(double)acc*4/t/1e9,(double)acc*64/t/1e9,(double)sink);
    free(src);
}

// THP 开关：本机 transparent_hugepage=always，但 torch 的 caching allocator 复用
// 的页不一定拿到大页。NOHP=1 时显式关掉大页，量的是"4KB 页 + dTLB miss"的代价。
static void thp_policy(void *p,size_t size){
    const char *e=getenv("NOHP");
    madvise(p,size,(e&&e[0]=='1')?MADV_NOHUGEPAGE:MADV_HUGEPAGE);
}

static void gather_mt(size_t size,uint64_t acc,int nt,int mode){
    size_t n=size/4; float *src=aligned_alloc(64,size);
    thp_policy(src,size);
    for(size_t i=0;i<n;i++) src[i]=(float)(i&1023);
    uint64_t m=n-1;
    uint64_t per=acc/(uint64_t)nt; if(per<8) per=8;
    double best=1e30, t=0;      // shared：single 里写、所有线程读
    #pragma omp parallel num_threads(nt)
    {
        int tid=omp_get_thread_num();
        uint64_t g=0x243F6A8885A308D3ULL+size+mode*131+tid;
        uint64_t s0=splitmix(&g),s1=splitmix(&g),s2=splitmix(&g),s3=splitmix(&g);
        volatile float sink=0;
        for(int r=0;r<3;r++){
            #pragma omp barrier
            #pragma omp single
            { t=now(); }
            #pragma omp barrier
            if(mode==0){
                for(uint64_t i=0;i+3<per;i+=4){
                    sink+=src[((i*16)&m)]; sink+=src[(((i+1)*16)&m)];
                    sink+=src[(((i+2)*16)&m)]; sink+=src[(((i+3)*16)&m)];
                }
            } else {
                for(uint64_t i=0;i+3<per;i+=4){
                    sink+=src[xs(&s0)&m]; sink+=src[xs(&s1)&m];
                    sink+=src[xs(&s2)&m]; sink+=src[xs(&s3)&m];
                }
            }
            #pragma omp barrier
            #pragma omp single
            { double d=now()-t; if(d<best) best=d; }
        }
        if(sink==1e30f) src[0]=sink;
    }
    printf("gathermt size=%zuMB threads=%d mode=%s acc/thread=%.0f ns/access=%.2f line_GBps=%.2f\n",
        size/1024/1024,nt,mode?"random":"seq",(double)per,
        1e9*best/(double)per,(double)per*(double)nt*64/best/1e9);
    free(src);
}

// 行随机 gather：每线程读 ROWS/nt 条随机行（A 个 float 连续），顺序写回。
// 这是 rowcopy 类路径的硬件上限：读侧是随机 64B line，写侧是连续 A*4 字节。
#define C4(d0,c,A,p0,p1,p2,p3) do { \
    (d0)[c]=(p0)[c]; (d0)[(A)+(c)]=(p1)[c]; \
    (d0)[2*(A)+(c)]=(p2)[c]; (d0)[3*(A)+(c)]=(p3)[c]; } while(0)

// 带下标的 gather：地址来自"顺序读的 int64 下标数组"，与真实内核同构
// （load idx -> load data 的依赖链）；CHAINS 控制同时算几个地址再发 load，
// 用来把"每核在飞 miss 条数"和"下标读取/检查"分开量。
// 直接量"每核同时在飞多少条 miss"：S 条**互相独立**的指针追逐链。
// 每条链内部是依赖的（后一个地址来自前一个值），所以一条链最多 1 条在飞；
// S 条链并行 → 在飞条数 = S（直到被带宽或 MSHR 数量卡住）。
// ns/access = 总时间 / (访问次数 × S)，能直接读出"成本 ~ 延迟/S"能到哪。
static void mlp(size_t size,int streams,uint64_t acc){
    uint64_t n=size/8;
    uint64_t *p=aligned_alloc(64,size);
    thp_policy(p,size);
    for(uint64_t i=0;i<n;i++) p[i]=i;
    uint64_t s=88172645463325252ULL+streams;
    for(uint64_t i=n-1;i>0;i--){uint64_t j=xs(&s)%(i+1);uint64_t t=p[i];p[i]=p[j];p[j]=t;}
    uint64_t cur[64];
    for(int k=0;k<streams;k++) cur[k]=xs(&s)%n;
    uint64_t iters=acc/streams, nacc=iters*(uint64_t)streams;
    double t0=now();
    for(uint64_t i=0;i<iters;i++){
        for(int k=0;k<streams;k++) cur[k]=p[cur[k]];
    }
    double t=now()-t0;
    uint64_t sinkv=0; for(int k=0;k<streams;k++) sinkv+=cur[k];
    printf("mlp size=%zuMB streams=%d accesses=%lu ns/access=%.2f line_GBps=%.2f sink=%lu\n",
        size/1024/1024,streams,nacc,1e9*t/(double)nacc,
        (double)nacc*64/t/1e9,(unsigned long)sinkv);
    free(p);
}

// chains 必须是 1/2/4/8/16/32：每个分支写死**恰好 chains 条**独立 load，
// 用 4 个累加器把加法链摊开（单累加器会变成 4 cycle 的依赖链，掩盖访存）。
#define Q4(o) a0+=src[p[o]]; a1+=src[p[o+1]]; a2+=src[p[o+2]]; a3+=src[p[o+3]];
#define LOOPN(N,BODY) \
    for(uint64_t i=0;i+(N)<=per;i+=(N)){ \
        const int64_t *p=idx+(i&(idxn-1)); \
        float a0=0,a1=0,a2=0,a3=0; BODY \
        sink+=a0+a1+a2+a3; }

static void idxgather(size_t size,uint64_t acc,int nt,int chains){
    size_t n=size/4; float *src=aligned_alloc(64,size);
    thp_policy(src,size);
    for(size_t i=0;i<n;i++) src[i]=(float)(i&1023);
    // 尾部多复制 32 项：内层一次读 chains 个下标时不会越过数组末尾，
    // 于是掩码可以直接用 (idxn-1)，每轮都真的前进一整批。
    uint64_t idxn=1<<20; int64_t *idx=malloc((idxn+64)*sizeof(int64_t));
    uint64_t g=0x2545F4914F6CDD1DULL+size;
    for(uint64_t i=0;i<idxn;i++) idx[i]=(int64_t)(splitmix(&g)%n);
    for(uint64_t i=idxn;i<idxn+64;i++) idx[i]=idx[i&(idxn-1)];
    uint64_t per=acc/(uint64_t)nt; if(per<idxn) per=idxn;
    per=(per/chains)*chains;
    double best=1e30, t=0;
    #pragma omp parallel num_threads(nt)
    {
        volatile float sink=0;
        for(int r=0;r<3;r++){
            #pragma omp barrier
            #pragma omp single
            { t=now(); }
            #pragma omp barrier
            // 两段式：先把 chains 个地址（顺序读的下标数组）算出来，再一口气发
            // chains 条数据 load。真实内核要做的正是这件事。
            switch(chains){
            case 1:  LOOPN(1,  a0+=src[p[0]];) break;
            case 2:  LOOPN(2,  a0+=src[p[0]]; a1+=src[p[1]];) break;
            case 4:  LOOPN(4,  Q4(0)) break;
            case 8:  LOOPN(8,  Q4(0) Q4(4)) break;
            case 16: LOOPN(16, Q4(0) Q4(4) Q4(8) Q4(12)) break;
            default: LOOPN(32, Q4(0) Q4(4) Q4(8) Q4(12) Q4(16) Q4(20) Q4(24) Q4(28)) break;
            }
            #pragma omp barrier
            #pragma omp single
            { double d=now()-t; if(d<best) best=d; }
        }
        if(sink==1e30f) src[0]=sink;
    }
    double nacc=(double)per;
    printf("idxgather size=%zuMB threads=%d chains=%d acc/thread=%.0f "
        "ns/access=%.2f line_GBps=%.2f\n",
        size/1024/1024,nt,chains,nacc,1e9*best/nacc,
        nacc*(double)nt*64/best/1e9);
    free(idx); free(src);
}

static void rowgather(size_t size,int A,int nt,uint64_t rows,int mode){
    // mode: 0=内联逐元素（1 行/次） 1=4 行流水 2=memcpy 调用（运行时长度）
    size_t n=size/4; float *src=aligned_alloc(64,size);
    thp_policy(src,size);
    for(size_t i=0;i<n;i++) src[i]=(float)(i&1023);
    uint64_t nrows=n/(uint64_t)A;
    uint64_t rmask=1; while(rmask*2<=nrows) rmask*=2; rmask-=1;  // 2 的幂掩码，避免 64 位除法进内循环
    uint64_t per=rows/(uint64_t)nt; if(per<8) per=8;
    // 每线程独立输出区：共用一个 dst 会让 36 个线程写同一批 line（伪共享 +
    // coherence 流量），测出来的不是内存系统的上限。
    float *dst=aligned_alloc(64,(size_t)nt*per*A*sizeof(float)+64);
    double best=1e30, t=0;      // shared
    #pragma omp parallel num_threads(nt)
    {
        int tid=omp_get_thread_num();
        uint64_t g=0x9E3779B97F4A7C15ULL+size+A*17+tid;
        uint64_t s0=splitmix(&g),s1=splitmix(&g);
        for(int r=0;r<3;r++){
            #pragma omp barrier
            #pragma omp single
            { t=now(); }
            #pragma omp barrier
            // 4 行流水：把 4 条随机行的地址同时算出来、再逐列交错拷贝，让 4 条
            // line 同时在飞——真实行拷贝内核（或 OoO 展开）的形态。只发 1 行时
            // 每线程在飞请求数=1，测出来的是延迟而不是带宽上限。
            if(mode>=3){
                // 预取版：地址提前 D 行算好放进环形缓冲，对第 i+D 行发 prfm，
                // 再拷第 i 行。D=8 是 L1 keep，D=16 是 L2 keep + 更远距离。
                const int D = (mode==3)?8:16;
                uint64_t ring[16];
                for(int d=0;d<D;d++) ring[d]=xs(&s0)&rmask;
                for(uint64_t i=0;i<per;i++){
                    uint64_t row=ring[i&(D-1)];
                    uint64_t nxt=xs(&s0)&rmask;
                    ring[i&(D-1)]=nxt;
                    const float *pf=src+nxt*(uint64_t)A;
                    asm volatile("prfm pldl1keep, [%0]" :: "r"(pf) : "memory");
                    const float *sp=src+row*(uint64_t)A;
                    float *dp=dst+((uint64_t)tid*per+i)*(uint64_t)A;
                    for(int c=0;c<A;c++) dp[c]=sp[c];
                }
                #pragma omp barrier
                #pragma omp single
                { double d=now()-t; if(d<best) best=d; }
                continue;
            }
            uint64_t i=0;
            for(;mode==1 && i+3<per;i+=4){
                const float *p0=src+(xs(&s0)&rmask)*(uint64_t)A;
                const float *p1=src+(xs(&s1)&rmask)*(uint64_t)A;
                const float *p2=src+(xs(&s0)&rmask)*(uint64_t)A;
                const float *p3=src+(xs(&s1)&rmask)*(uint64_t)A;
                float *d0=dst+((uint64_t)tid*per+i)*(uint64_t)A;
                for(int c=0;c<A;c++){
                    C4(d0,c,A,p0,p1,p2,p3);
                }
            }
            for(;i<per;i++){
                uint64_t row=xs(&s0)&rmask;
                const float *sp=src+row*(uint64_t)A;
                float *dp=dst+((uint64_t)tid*per+i)*(uint64_t)A;
                if(mode==2) memcpy(dp,sp,(size_t)A*sizeof(float));
                else for(int c=0;c<A;c++) dp[c]=sp[c];
            }
            #pragma omp barrier
            #pragma omp single
            { double d=now()-t; if(d<best) best=d; }
        }
        if(dst[0]==1e30f) src[0]=dst[0];
    }
    // 校验"每行真的被拷过"：dst 是独立的每线程输出区，若拷贝被优化掉，
    // 校验和会退化成 0（或常数），这时候上面的 ns/row 不能信。
    double chk=0;
    for(size_t i=0;i<(size_t)nt*per*(size_t)A;i++) chk+=dst[i];
    double rowb=(double)A*4;
    int lb=(int)((rowb+63)/64);
    printf("rowgather mode=%d size=%zuMB A=%d rowB=%.0f t=%d rows/thread=%.0f "
        "ns/row=%.2f useful_GBps=%.2f line_GBps=%.2f chk=%.1f\n",
        mode,size/1024/1024,A,rowb,nt,(double)per,
        1e9*best/(double)per,2.0*(double)per*(double)nt*rowb/best/1e9,
        (double)per*(double)nt*lb*64/best/1e9,chk);
    free(dst); free(src);
}

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: %s latency SIZE_KB ACC | bandwidth SIZE_MB THREADS RD|CP [REPS] | gather SIZE_MB ACC seq|s16|rand | gathermt SIZE_MB ACC THREADS seq|rand | rowgather SIZE_MB A THREADS ROWS [pipe] | mlp SIZE_MB STREAMS ACC | idxgather SIZE_MB ACC THREADS CHAINS\n",argv[0]);return 1;}
    if(!strcmp(argv[1],"latency")) latency(atol(argv[2])*1024ULL,strtoull(argv[3],0,10));
    else if(!strcmp(argv[1],"bandwidth")) bandwidth(atol(argv[2])*1024ULL*1024,atoi(argv[3]),!strcmp(argv[4],"cp"),argc>5?atoi(argv[5]):1);
    else if(!strcmp(argv[1],"gathermt")) gather_mt(atol(argv[2])*1024ULL*1024,strtoull(argv[3],0,10),atoi(argv[4]),!strcmp(argv[5],"rand")?1:0);
    else if(!strcmp(argv[1],"mlp")) mlp(atol(argv[2])*1024ULL*1024,atoi(argv[3]),strtoull(argv[4],0,10));
    else if(!strcmp(argv[1],"idxgather")) idxgather(atol(argv[2])*1024ULL*1024,strtoull(argv[3],0,10),atoi(argv[4]),atoi(argv[5]));
    else if(!strcmp(argv[1],"rowgather")) {
        int mode=0;
        if(argc>6){
            if(!strcmp(argv[6],"pipe")) mode=1;
            else if(!strcmp(argv[6],"memcpy")) mode=2;
            else if(!strcmp(argv[6],"pf8")) mode=3;
            else if(!strcmp(argv[6],"pf16")) mode=4;
        }
        rowgather((size_t)(atof(argv[2])*1048576.0),atoi(argv[3]),atoi(argv[4]),
                  strtoull(argv[5],0,10),mode);
    }
    else if(!strcmp(argv[1],"gather")){
        int m=!strcmp(argv[4],"seq")?0:!strcmp(argv[4],"s16")?1:2;
        gather(atol(argv[2])*1024ULL*1024,strtoull(argv[3],0,10),m);
    }
    return 0;
}
