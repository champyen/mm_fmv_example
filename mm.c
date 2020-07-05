#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define TEST_M 512
#define TEST_K 512
#define TEST_N 512


#define TSIZE 8

#if defined (__clang__)
typedef float vfloat __attribute__((ext_vector_type(TSIZE)));
#else
typedef float vfloat __attribute__ ((vector_size (TSIZE*4)));
#endif

#ifdef ENABLE_VEC
__attribute__ ((target_clones ("avx512f", "avx2", "avx", "sse4.2", "default")))
#endif
void gemm_vec(float *a, int sa, float *b, int sb, float *c, int sc)
{
#ifdef ENABLE_VEC
    vfloat vb[TSIZE];
	for(int y = 0; y < TSIZE; y++){
	    vb[y] = *((vfloat*)(b + sb*y));
	}
	for(int y = 0; y < TSIZE; y++){
	    vfloat vc = *((vfloat*)(c + sc*y));
	    vfloat va = *((vfloat*)(a + sa*y));
		for(int x = 0; x < TSIZE; x++){
            vc += va[x] * vb[x];
		}
		*((vfloat*)(c + sc*y)) = vc;
	}
#else
	for(int y = 0; y < TSIZE; y++){
		for(int x = 0; x < TSIZE; x++){
			float fc = *(c + sc*y + x);
			for(int k = 0; k < TSIZE; k++){
				fc += a[sa*y + k] * b[sb*k + x];
			}
			*(c + sc*y + x) = fc;
		}
	}
#endif
}

int main(void)
{
	float* ma = (float*)aligned_alloc(256, sizeof(float)*TEST_K*TEST_M);
	float* mb = (float*)aligned_alloc(256, sizeof(float)*TEST_N*TEST_K);
	float* mc = (float*)aligned_alloc(256, sizeof(float)*TEST_N*TEST_M);
	float* chk = (float*)aligned_alloc(256, sizeof(float)*TEST_N*TEST_M);

	for(int y = 0; y < TEST_M; y++){
		for(int x = 0; x < TEST_K; x++){
			ma[y*TEST_K + x] = (float)(rand()%256/256.0);
		}
	}
	for(int y = 0; y < TEST_K; y++){
		for(int x = 0; x < TEST_N; x++){
			mb[y*TEST_N + x] = (float)(rand()%256/256.0);
		}
	}
	for(int y = 0; y < TEST_M; y++){
		for(int x = 0; x < TEST_N; x++){
			mc[y*TEST_N + x] = (float)0.0;
			chk[y*TEST_N + x] = (float)0.0;
		}
	}

	struct timeval stime, etime;
	gettimeofday(&stime, NULL);

    //parallel here
    #pragma omp parallel for
	for(int m = 0; m < TEST_M; m+=TSIZE){
		for(int n = 0; n < TEST_N; n+=TSIZE){
			for(int k = 0; k < TEST_K; k+=TSIZE){
				gemm_vec(
							ma + m*TEST_K + k, TEST_K,
							mb + k*TEST_N + n, TEST_N,
							mc + m*TEST_N + n, TEST_N
						);
			}
		}
	}
	gettimeofday(&etime, NULL);
	printf("FP32 SIMD: %ld us\n", (etime.tv_sec - stime.tv_sec)*1000000 + (etime.tv_usec - stime.tv_usec));

	gettimeofday(&stime, NULL);
	for(int m = 0; m < TEST_M; m++){
		for(int n = 0; n < TEST_N; n++){
			float val = 0.0;
			for(int k = 0; k < TEST_K; k++){
				val += ma[m*TEST_K + k]*mb[k*TEST_N+n];
			}
			chk[m*TEST_N + n] = val;
		}
	}
	gettimeofday(&etime, NULL);
	printf("NAIVE: %ld us\n", (etime.tv_sec - stime.tv_sec)*1000000 + (etime.tv_usec - stime.tv_usec));

	for(int m = 0; m < TEST_M; m++){
		for(int n = 0; n < TEST_N; n++){
			float val = chk[m*TEST_N + n] - mc[m*TEST_N + n];
			if( fabs(val) > 0.1){
				printf("(%d,%d), %f %f\n", m, n, chk[m*TEST_N + n], mc[m*TEST_N + n]);
			}
		}
	}

	printf("DONE!\n");
	return 0;
}
