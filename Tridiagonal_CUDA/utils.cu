#include <math.h>
#include <cuda.h>
#include "utils.h"

template<typename val_t>
__forceinline__ __device__ val_t warp_sum(val_t val) {
    for(int i = 16; i >= 1; i = i >> 1) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val;
}

template<int THREAD_NUM>
__forceinline__ __device__ int min_threads(int size) {
    if(size > (THREAD_NUM >> 1)) {
        return THREAD_NUM >> 1;
    } else {
        int x = size - 1;
        x |= (x >>  1);
        x |= (x >>  2);
        x |= (x >>  4);
        x |= (x >>  8);
        return x ^ (x >> 1);
    }
}

/* Treat NaN specially when calculating maximum */
template<typename val_t, int THREAD_NUM>
__global__ void check_answer_first_kernel(int size, const val_t *ans0, const val_t *ans1, \
                            val_t *norm_1_out, val_t *norm_2_out, val_t *norm_inf_out, \
                            int *maxid_out) {
    __shared__ val_t norm_1_shmem[THREAD_NUM], norm_2_shmem[THREAD_NUM], norm_inf_shmem[THREAD_NUM];
    __shared__ int maxid_shmem[THREAD_NUM];
    const int gid = threadIdx.x + blockDim.x * blockIdx.x;
    const int gsize = blockDim.x * gridDim.x;
    val_t err0 = (gid + 0*gsize < size) ? (ans0[gid + 0*gsize] - ans1[gid + 0*gsize]) : 0.0;
    val_t err1 = (gid + 1*gsize < size) ? (ans0[gid + 1*gsize] - ans1[gid + 1*gsize]) : 0.0;
    val_t err2 = (gid + 2*gsize < size) ? (ans0[gid + 2*gsize] - ans1[gid + 2*gsize]) : 0.0;
    val_t err3 = (gid + 3*gsize < size) ? (ans0[gid + 3*gsize] - ans1[gid + 3*gsize]) : 0.0;
    val_t norm_1 = (abs(err0) + abs(err1)) + (abs(err2) + abs(err3));
    val_t norm_2 = (err0 * err0 + err1 * err1) + (err2 * err2 + err3 * err3);
    val_t norm_inf = err0;
    int maxid = gid;
    if((norm_inf < err1) || isnan(err1)) { norm_inf = err1; maxid = gid + gsize; }
    if((norm_inf < err2) || isnan(err2)) { norm_inf = err2; maxid = gid + 2*gsize; }
    if((norm_inf < err3) || isnan(err3)) { norm_inf = err3; maxid = gid + 3*gsize; }
    norm_1_shmem[threadIdx.x] = norm_1;
    norm_2_shmem[threadIdx.x] = norm_2;
    norm_inf_shmem[threadIdx.x] = norm_inf;
    maxid_shmem[threadIdx.x] = maxid;
    __syncthreads();
    for(int half = min_threads<THREAD_NUM>(size); half >= 32; half = half >> 1) {
        if(threadIdx.x < half) {
            norm_1_shmem[threadIdx.x] += norm_1_shmem[threadIdx.x + half];
            norm_2_shmem[threadIdx.x] += norm_2_shmem[threadIdx.x + half];
            if((norm_inf_shmem[threadIdx.x] < norm_inf_shmem[threadIdx.x + half]) \
                 || isnan(norm_inf_shmem[threadIdx.x + half])) {
                norm_inf_shmem[threadIdx.x] = norm_inf_shmem[threadIdx.x + half];
                maxid_shmem[threadIdx.x] = maxid_shmem[threadIdx.x + half];
            }
        } else return;
        __syncthreads();
    }
    if(threadIdx.x < 32) {
        norm_1 = warp_sum(norm_1_shmem[threadIdx.x]);
        norm_2 = warp_sum(norm_2_shmem[threadIdx.x]);
        norm_inf = norm_inf_shmem[threadIdx.x];
        maxid = maxid_shmem[threadIdx.x];
        for(int i = 16; i >= 1; i = i >> 1) {
            val_t v = __shfl_xor_sync(0xffffffff, norm_inf, i);
            int ii = __shfl_xor_sync(0xffffffff, maxid, i);
            if((norm_inf < v)  || isnan(v)) {
                norm_inf = v;
                maxid = ii;
            }
        }
    }
    
    if(threadIdx.x == 0) {
        norm_1_out[blockIdx.x] = norm_1;
        norm_2_out[blockIdx.x] = norm_2;
        norm_inf_out[blockIdx.x] = norm_inf;
        maxid_out[blockIdx.x] = maxid;
    }
}

template<typename val_t, int THREAD_NUM>
__global__ void check_answer_rest_kernel(int size, const val_t *norm_1_in, const val_t *norm_2_in, \
                            const val_t *norm_inf_in, const int *maxid_in, val_t *norm_1_out, \
                            val_t *norm_2_out, val_t *norm_inf_out, int *maxid_out) {
    __shared__ val_t norm_1_shmem[THREAD_NUM], norm_2_shmem[THREAD_NUM], norm_inf_shmem[THREAD_NUM];
    __shared__ int maxid_shmem[THREAD_NUM];
    const int gid = threadIdx.x + blockDim.x * blockIdx.x;
    const int gsize = blockDim.x * gridDim.x;
    val_t t0 = (gid + 0*gsize < size) ? norm_1_in[gid + 0*gsize] : 0.0;
    val_t t1 = (gid + 1*gsize < size) ? norm_1_in[gid + 1*gsize] : 0.0;
    val_t t2 = (gid + 2*gsize < size) ? norm_1_in[gid + 2*gsize] : 0.0;
    val_t t3 = (gid + 3*gsize < size) ? norm_1_in[gid + 3*gsize] : 0.0;
    val_t norm_1 = (t0 + t1) + (t2 + t3);
    t0 = (gid + 0*gsize < size) ? norm_2_in[gid + 0*gsize] : 0.0;
    t1 = (gid + 1*gsize < size) ? norm_2_in[gid + 1*gsize] : 0.0;
    t2 = (gid + 2*gsize < size) ? norm_2_in[gid + 2*gsize] : 0.0;
    t3 = (gid + 3*gsize < size) ? norm_2_in[gid + 3*gsize] : 0.0;
    val_t norm_2 = (t0 + t1) + (t2 + t3);
    val_t norm_inf = 0;
    int maxid;
    if((gid + 0*gsize < size) && ((norm_inf < norm_inf_in[gid + 0*gsize]) \
            || isnan(norm_inf_in[gid + 0*gsize]))) {
        norm_inf = norm_inf_in[gid];
        maxid    = maxid_in   [gid];
    }
    if((gid + 1*gsize < size) && ((norm_inf < norm_inf_in[gid + 1*gsize]) \
            || isnan(norm_inf_in[gid + 1*gsize]))) {
        norm_inf = norm_inf_in[gid + 1*gsize];
        maxid = maxid_in[gid + 1*gsize];
    }
    if((gid + 2*gsize < size) && ((norm_inf < norm_inf_in[gid + 2*gsize]) \
            || isnan(norm_inf_in[gid + 2*gsize]))) {
        norm_inf = norm_inf_in[gid + 2*gsize];
        maxid = maxid_in[gid + 2*gsize];
    }
    if((gid + 3*gsize < size) && ((norm_inf < norm_inf_in[gid + 3*gsize]) \
            || isnan(norm_inf_in[gid + 3*gsize]))) {
        norm_inf = norm_inf_in[gid + 3*gsize];
        maxid = maxid_in[gid + 3*gsize];
    }
    norm_1_shmem[threadIdx.x] = norm_1;
    norm_2_shmem[threadIdx.x] = norm_2;
    norm_inf_shmem[threadIdx.x] = norm_inf;
    maxid_shmem[threadIdx.x] = maxid;
    __syncthreads();
    for(int half = min_threads<THREAD_NUM>(size); half >= 32; half = half >> 1) {
        if(threadIdx.x < half) {
            norm_1_shmem[threadIdx.x] += norm_1_shmem[threadIdx.x + half];
            norm_2_shmem[threadIdx.x] += norm_2_shmem[threadIdx.x + half];
            if((norm_inf_shmem[threadIdx.x] < norm_inf_shmem[threadIdx.x + half]) \
                    || isnan(norm_inf_shmem[threadIdx.x + half])) {
                norm_inf_shmem[threadIdx.x] = norm_inf_shmem[threadIdx.x + half];
                maxid_shmem[threadIdx.x] = maxid_shmem[threadIdx.x + half];
            }
        } else return;
        __syncthreads();
    }
    if(threadIdx.x < 32) {
        norm_1 = warp_sum(norm_1_shmem[threadIdx.x]);
        norm_2 = warp_sum(norm_2_shmem[threadIdx.x]);
        norm_inf = norm_inf_shmem[threadIdx.x];
        maxid = maxid_shmem[threadIdx.x];
        for(int i = 16; i >= 1; i = i >> 1) {
            val_t v = __shfl_xor_sync(0xffffffff, norm_inf, i);
            int ii = __shfl_xor_sync(0xffffffff, maxid, i);
            if((norm_inf < v) || isnan(v)) {
                norm_inf = v;
                maxid = ii;
            }
        }
    }
    
    if(threadIdx.x == 0) {
        norm_1_out[blockIdx.x] = norm_1;
        norm_2_out[blockIdx.x] = norm_2;
        norm_inf_out[blockIdx.x] = norm_inf;
        maxid_out[blockIdx.x] = maxid;
    }
}

inline int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}

template<typename val_t>
int check_impl(const val_t *ans0, const val_t *ans1_cpu, val_t *ans1, val_t *buffer, int *ibuffer, int size, val_t *stat) {
    int maxid;
    int reduce_size = ceiling(size, 4096);
    int current_size = size;
    val_t *norm_1_a = buffer;
    val_t *norm_2_a = buffer + reduce_size;
    val_t *norm_inf_a = buffer + reduce_size * 2;
    val_t *norm_1_b = buffer + reduce_size * 3;
    val_t *norm_2_b = buffer + reduce_size * 4;
    val_t *norm_inf_b = buffer + reduce_size * 5;
    val_t *tmp;
    int *maxid_a = ibuffer;
    int *maxid_b = ibuffer + reduce_size;
    int *tmp1;
    cudaMemcpy(ans1, ans1_cpu, sizeof(val_t) * size, cudaMemcpyHostToDevice);
    check_answer_first_kernel<val_t, 1024><<<reduce_size, 1024>>>( \
            size, ans0, ans1, norm_1_a, norm_2_a, norm_inf_a, maxid_a);
    current_size = reduce_size;
    while(current_size > 1) {
        int reduce_size = ceiling(current_size, 4096);
        check_answer_rest_kernel<val_t, 1024><<<reduce_size, 1024>>>(current_size, \
            norm_1_a, norm_2_a, norm_inf_a, maxid_a, norm_1_b, norm_2_b, norm_inf_b, maxid_b);
        tmp = norm_1_a; norm_1_a = norm_1_b; norm_1_b = tmp;
        tmp = norm_2_a; norm_2_a = norm_2_b; norm_2_b = tmp;
        tmp = norm_inf_a; norm_inf_a = norm_inf_b; norm_inf_b = tmp;
        tmp1 = maxid_a; maxid_a = maxid_b; maxid_b = tmp1;
        current_size = reduce_size;
    }
    cudaMemcpy(&stat[0], norm_1_a, sizeof(val_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&stat[1], norm_2_a, sizeof(val_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&stat[2], norm_inf_a, sizeof(val_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxid, maxid_a, sizeof(int), cudaMemcpyDeviceToHost);
    stat[0] = stat[0] / size;
    stat[1] = sqrt(stat[1] / size);
    return maxid;
}

int check(float *ans0, const float *ans1, float *buffer0, float *buffer, int *ibuffer, int size, float *stat, int layout) {
    if(layout == 0) {
        return check_impl<float>(ans0, ans1, buffer0, buffer, ibuffer, size, stat);
    } else {
        return check_impl<float>(buffer0 + 3 * size, ans1, ans0, buffer, ibuffer, size, stat);
    }
}

int check(double *ans0, const double *ans1, double *buffer0, double *buffer, int *ibuffer, int size, double *stat, int layout) {
    return check_impl<double>(ans0, ans1, buffer0, buffer, ibuffer, size, stat);
}

void *cuda_malloc(size_t size) {
    void *p;
    int ret = cudaMalloc(&p, size);
    return (ret == cudaSuccess) ? p : NULL;
}

void cuda_free(void *p) {
    cudaFree(p);
}

template<typename val_t>
__global__ void init_batch_kernel(val_t *a, val_t *b, val_t *c, val_t *d, int size, int batch) {
    val_t *src;
    val_t *dst;
    switch (threadIdx.y) {
    case 0:
        src = a + threadIdx.x;
        break;
    case 1:
        src = b + threadIdx.x;
        break;
    case 2:
        src = c + threadIdx.x;
        break;
    case 3:
        src = d + threadIdx.x;
    }
    dst = src + (1 + blockIdx.x) * size;
    for(int i = 0; i < size; i += blockDim.x) {
        dst[i] = src[i];
    }
}

template<typename val_t>
__global__ void init_batch_kernel_2(val_t *data, int size, int batch) {
    val_t *dst = data + (1 + blockIdx.x) * size * 4;
    for(int i = 0; i < size; i += blockDim.x) {
        dst[i] = data[i];
    }
}

template<typename val_t>
void init_batch_cusparse(const val_t *a, const val_t *b, const val_t *c, const val_t *d, \
                val_t *a_bat, val_t *b_bat, val_t *c_bat, val_t *d_bat, int size, int batch) {
    cudaMemcpy(a_bat, a, sizeof(val_t) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_bat, b, sizeof(val_t) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_bat, c, sizeof(val_t) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bat, d, sizeof(val_t) * size, cudaMemcpyHostToDevice);
    init_batch_kernel<val_t><<<batch - 1, dim3(32, 4, 1)>>>(a_bat, b_bat, c_bat, d_bat, size, batch);
    cudaDeviceSynchronize();
}


template<typename val_t>
void init_batch_BPLG(const val_t *a, const val_t *b, const val_t *c, const val_t *d, \
                val_t *data, int size, int batch) {
    cudaMemcpy(data + 0 * size, a, sizeof(val_t) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(data + 1 * size, b, sizeof(val_t) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(data + 2 * size, c, sizeof(val_t) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(data + 3 * size, d, sizeof(val_t) * size, cudaMemcpyHostToDevice);
    init_batch_kernel_2<val_t><<<batch - 1, 128>>>(data, size, batch);
    cudaDeviceSynchronize();
}

void init_batch(const float *a, const float *b, const float *c, const float *d, \
                float *a_bat, float *b_bat, float *c_bat, float *d_bat, int size, int batch, int layout) {
    if(layout == 0) {
        init_batch_cusparse<float>(a, b, c, d, a_bat, b_bat, c_bat, d_bat, size, batch);
    } else /*if(layout == 1)*/ {
        init_batch_BPLG<float>(a, b, c, d, a_bat, size, batch);
    }
}

void init_batch(const double *a, const double *b, const double *c, const double *d, \
                double *a_bat, double *b_bat, double *c_bat, double *d_bat, int size, int batch, int layout) {
    init_batch_cusparse<double>(a, b, c, d, a_bat, b_bat, c_bat, d_bat, size, batch);
}