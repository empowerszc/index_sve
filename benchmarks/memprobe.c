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

static void bandwidth(size_t size,int nt,int wr){
    size_t n=size/8; uint64_t *a=aligned_alloc(64,size); uint64_t *b=wr?aligned_alloc(64,size):NULL;
    for(size_t i=0;i<n;i++) a[i]=i*3+1;
    if(wr) memset(b,0,size);
    double t0=now();
    #pragma omp parallel num_threads(nt)
    {
        size_t lo=(size_t)((double)n*omp_get_thread_num()/nt), hi=(size_t)((double)n*(omp_get_thread_num()+1)/nt);
        if(!wr){uint64_t s0=0,s1=0,s2=0,s3=0;
            for(size_t i=lo;i+3<hi;i+=4){s0+=a[i];s1+=a[i+1];s2+=a[i+2];s3+=a[i+3];}
            if(omp_get_thread_num()==0) a[0]+=s0+s1+s2+s3;
        }else{
            for(size_t i=lo;i<hi;i++) b[i]=a[i]*3+1;
            if(omp_get_thread_num()==0) b[0]+=1;
        }
    }
    double t=now()-t0;
    double bytes=(double)size*(wr?2:1);
    printf("bandwidth size=%zuMB threads=%d op=%s GBps=%.2f\n",size/1024/1024,nt,wr?"copy":"read",bytes/t/1e9);
    if(wr && b[0]==0xdeadbeef) printf("x");
    free(a); free(b);
}

static void gather(size_t size,uint64_t acc,int mode){
    size_t n=size/4; float *src=aligned_alloc(64,size);
    for(size_t i=0;i<n;i++) src[i]=(float)(i&1023);
    uint64_t m=n-1; uint64_t s0=1,s1=2,s2=3,s3=4,s4=5,s5=6,s6=7,s7=8;
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

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: %s latency SIZE_KB ACC | bandwidth SIZE_MB THREADS RD|CP | gather SIZE_MB ACC seq|s16|rand\n",argv[0]);return 1;}
    if(!strcmp(argv[1],"latency")) latency(atol(argv[2])*1024ULL,strtoull(argv[3],0,10));
    else if(!strcmp(argv[1],"bandwidth")) bandwidth(atol(argv[2])*1024ULL*1024,atoi(argv[3]),!strcmp(argv[4],"cp"));
    else if(!strcmp(argv[1],"gather")){
        int m=!strcmp(argv[4],"seq")?0:!strcmp(argv[4],"s16")?1:2;
        gather(atol(argv[2])*1024ULL*1024,strtoull(argv[3],0,10),m);
    }
    return 0;
}
