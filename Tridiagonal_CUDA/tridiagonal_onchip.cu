#include<iostream>
#include<cuda.h>
#include"tridiagonal.h"

#define CUDA_CHECK_ERROR(errorMessage) {                              \
    cudaError_t err = cudaGetLastError();                             \
    if( cudaSuccess != err) {                                         \
        std::cerr << "Cuda error: " << errorMessage << " in file '"   \
            << __FILE__ << "' in line " << __LINE__ << " : "          \
            << cudaGetErrorString(err) << "." << std::endl;           \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

#define ZERO(type) static_cast<type>(0)  /* otherwise nvcc use cvt instructions */
#define ONE(type) static_cast<type>(1)   /* otherwise nvcc use cvt instructions */

__device__ __constant__ int offset_table[14];
__device__ __constant__ int length_table[14];

inline void set_table(const int *lengths, const int *offsets, int n) {
    cudaMemcpyToSymbol(length_table, lengths, sizeof(int)*(n+1));
    cudaMemcpyToSymbol(offset_table, offsets, sizeof(int)*(n+2));
}

inline int padding(int n) {
    const int padding_table[32] = {\
        2,1,0,0,0,0,0,0,0,0,\
        0,0,0,0,0,0,0,0,0,0,\
        0,0,0,0,0,0,0,0,0,0,\
        0,3};
    return (n > 32) ? padding_table[n % 32] : 0;
}



template<typename val_t>
__global__ void CR_shmem_kernel(val_t *a_gbl, val_t *b_gbl, val_t *c_gbl, val_t *d_gbl, int n_eqt, int n_batch, int shmem_size, int iter_num)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    extern __shared__ double shared[];
    val_t *a_global = a_gbl + bid*n_eqt;
    val_t *b_global = b_gbl + bid*n_eqt;
    val_t *c_global = c_gbl + bid*n_eqt;
    val_t *d_global = d_gbl + bid*n_eqt;
    val_t* a_shared = reinterpret_cast<val_t*>(shared);
    val_t* b_shared = (val_t*)&a_shared[shmem_size];
    val_t* c_shared = (val_t*)&b_shared[shmem_size];
    val_t* d_shared = (val_t*)&c_shared[shmem_size];
    int n = n_eqt, n_even = n_eqt / 2, n_odd = n - n_even;
    int offset0 = offset_table[0], offset1 = offset_table[1];
    a_shared[tid] = a_global[tid*2];
    b_shared[tid] = b_global[tid*2];
    c_shared[tid] = c_global[tid*2];
    d_shared[tid] = d_global[tid*2];
    a_shared[tid+offset1] = a_global[tid*2+1];
    b_shared[tid+offset1] = b_global[tid*2+1];
    c_shared[tid+offset1] = c_global[tid*2+1];
    d_shared[tid+offset1] = d_global[tid*2+1];
    __syncthreads();
    //CR Forward

    for(int iter = 0; iter < iter_num; ++iter) {
        int offset2 = offset_table[iter+2];
        val_t aa, bb, cc, dd;
        if(tid < n_even) {
            bool boundary = tid + 1 >= n_odd;
            val_t k1 = - a_shared[tid + offset1] / b_shared[tid + offset0];
            val_t k2 = - c_shared[tid + offset1] / (boundary ? 1 : b_shared[tid + offset0 + 1]);
            aa = a_shared[tid + offset0] * k1;
            bb = b_shared[tid + offset1] + c_shared[tid + offset0] * k1 \
                    + (boundary ? ZERO(val_t) : a_shared[tid + offset0 + 1]) * k2;
            cc = (boundary ? ZERO(val_t) : c_shared[tid + offset0 + 1]) * k2;
            dd = d_shared[tid + offset1] + d_shared[tid + offset0] * k1 \
                    + (boundary ? ZERO(val_t) : d_shared[tid + offset0 + 1]) * k2;
        }
        __syncthreads();
        if(tid < n_even) {
            int w_idx = tid/2 + ((tid % 2 == 0) ? offset1 : offset2);
            a_shared[w_idx] = aa;
            b_shared[w_idx] = bb;
            c_shared[w_idx] = cc;
            d_shared[w_idx] = dd;
        }
        __syncthreads();
        n = n_even;
        n_even = n / 2;
        n_odd = n - n_even;
        offset0 = offset1;
        offset1 = offset2;
    }
    if(tid == 0){
        if(n == 2) {
            val_t det = b_shared[offset0] * b_shared[offset1] - c_shared[offset0] * a_shared[offset1];
            val_t x0 = d_shared[offset0] * b_shared[offset1] - c_shared[offset0] * d_shared[offset1];
            val_t x1 = b_shared[offset0] * d_shared[offset1] - d_shared[offset0] * a_shared[offset1];
            d_shared[offset0] = x0 / det;
            d_shared[offset0+1] = x1 / det;
        } else {
            d_shared[offset0] = d_shared[offset0] / b_shared[offset0];
        }
    }
    __syncthreads();

    for(int iter = iter_num - 1; iter >= 0; --iter) {
        val_t dd0, dd1;
        //int offset2 = offset1;
        offset1 = offset0;
        offset0 = offset_table[iter];
        n_even = n;
        n = length_table[iter];
        n_odd = n - n_even;
        
        if(tid < n_odd) {
            dd1 = (tid >= n_even) ? ZERO(val_t) : d_shared[tid+offset1];
            dd0 = (d_shared[tid+offset0] \
                    - a_shared[tid+offset0] * ((tid == 0) ? ZERO(val_t) : d_shared[tid+offset1-1]) \
                    - c_shared[tid+offset0] * dd1) \
                    / b_shared[tid+offset0];
        }
        __syncthreads();
        if(iter == 0) {
            d_global[tid*2] = dd0;
            d_global[tid*2+1] = dd1;
        } else if(tid < n_odd) {
            d_shared[tid*2+offset0] = dd0;
            if(tid < n_even) {
                d_shared[tid*2+offset0+1] = dd1;
            }
        }
        __syncthreads();
    }
}


template<typename val_t>
void CR_shmem(val_t *a_gbl, val_t *b_gbl, val_t *c_gbl, val_t *d_gbl, int n_eqt, int n_batch) {
    int offset_table_host[14];
    int length_table_host[13];
    int n, iter, offset, shmem_size;
    int n_even, n_odd, n_last;
    n_even = n_eqt / 2;
    n_odd = n_eqt - n_even;
    n = n_even;
    offset = n_odd;// + padding(n_odd);
    length_table_host[0] = n_eqt;
    offset_table_host[0] = 0;
    offset_table_host[1] = offset;
    for(iter = 0; n > 1; ++iter, n = n / 2) {
        offset = offset + n;
        length_table_host[iter+1] = n;
        offset_table_host[iter+2] = offset;
        n_last = n;
    }
    if(n_last == 3) {
        length_table_host[iter+1] = 0;
        offset_table_host[iter+2] = offset;
        ++iter;
    }
    shmem_size = offset + n;
    set_table(length_table_host, offset_table_host, iter);
    CR_shmem_kernel<val_t><<<n_batch, n_eqt/2, shmem_size*4*sizeof(val_t)>>>(
        a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch, shmem_size, iter);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("CR_shmem_kernel");
}


#define DEF_VARS(iii) val_t ra##iii, rb##iii, rc##iii, rd##iii;
#define READ_VAR(iii, radix) {\
    ra##iii = a[tid * radix + iii];\
    rb##iii = b[tid * radix + iii];\
    rc##iii = c[tid * radix + iii];\
    rd##iii = d[tid * radix + iii];\
}
#define READ_VAR_COND(iii, radix) {\
    ra##iii = (tid * radix + iii < n_eqt) ? a[tid * radix + iii] : 0;\
    rb##iii = (tid * radix + iii < n_eqt) ? b[tid * radix + iii] : 1;\
    rc##iii = (tid * radix + iii < n_eqt) ? c[tid * radix + iii] : 0;\
    rd##iii = (tid * radix + iii < n_eqt) ? d[tid * radix + iii] : 0;\
}

#define WRITE_SHARED(iii) {\
    a_shared[tid] = ra##iii; \
    b_shared[tid] = rb##iii; \
    c_shared[tid] = rc##iii; \
    d_shared[tid] = rd##iii; \
}

#define READ_SHARED(iii) {\
    ra##iii = (tid == n - 1) ? ZERO(val_t) : a_shared[tid+1]; \
    rb##iii = (tid == n - 1) ? ONE(val_t)  : b_shared[tid+1]; \
    rc##iii = (tid == n - 1) ? ZERO(val_t) : c_shared[tid+1]; \
    rd##iii = (tid == n - 1) ? ZERO(val_t) : d_shared[tid+1]; \
}

#define REDUCE(iin, iii, iip) {\
    k1 = - ra##iii / rb##iin; \
    k2 = - rc##iii / rb##iip; \
    ra##iii = k1 * ra##iin; \
    rc##iii = k2 * rc##iip; \
    rb##iii = rb##iii + k1 * rc##iin + k2 * ra##iip; \
    rd##iii = rd##iii + k1 * rd##iin + k2 * rd##iip; \
}


#define SUBSTITUDE(iin, iii, iip) {\
    rd##iii = (rd##iii - ra##iii * rd##iin - rc##iii * rd##iip) / rb##iii;\
}
#define WRITE_VAR(iii, radix) {\
    d[tid * radix + iii] = rd##iii;\
}
#define WRITE_VAR_COND(iii, radix) if(tid * 8 + iii < n_eqt){\
    d[tid * radix + iii] = rd##iii;\
}

template<typename val_t>
__global__ void CR_reg8_kernel(val_t *a_gbl, val_t *b_gbl, val_t *c_gbl, val_t *d_gbl, int n_eqt, int n_batch, int shmem_size, int iter_num)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    val_t *a = a_gbl + bid*n_eqt;
    val_t *b = b_gbl + bid*n_eqt;
    val_t *c = c_gbl + bid*n_eqt;
    val_t *d = d_gbl + bid*n_eqt;
    extern __shared__ double shared[];
    val_t* a_shared = reinterpret_cast<val_t*>(shared);
    val_t* b_shared = (val_t*)&a_shared[shmem_size];
    val_t* c_shared = (val_t*)&b_shared[shmem_size];
    val_t* d_shared = (val_t*)&c_shared[shmem_size];
    int n = n_eqt / 8, n_even = n / 2, n_odd = n - n_even;
    int offset0 = offset_table[0], offset1 = offset_table[1];
    int w_idx;

    val_t k1, k2;
    DEF_VARS(boundary)
    DEF_VARS(0)
    DEF_VARS(1)
    DEF_VARS(2)
    DEF_VARS(3)
    DEF_VARS(4)
    DEF_VARS(5)
    DEF_VARS(6)
    DEF_VARS(7)

    READ_VAR(0, 8)
    READ_VAR(1, 8)
    READ_VAR(2, 8)
    READ_VAR(3, 8)
    READ_VAR(4, 8)
    READ_VAR(5, 8)
    READ_VAR(6, 8)
    READ_VAR(7, 8)

    WRITE_SHARED(0)
    __syncthreads();
    READ_SHARED(boundary)
    REDUCE(0, 1, 2)
    REDUCE(2, 3, 4)
    REDUCE(4, 5, 6)
    REDUCE(6, 7, boundary)
    __syncthreads();

    WRITE_SHARED(1)
    __syncthreads();
    READ_SHARED(boundary)
    REDUCE(1, 3, 5)
    REDUCE(5, 7, boundary)
    __syncthreads();

    WRITE_SHARED(3)
    __syncthreads();
    READ_SHARED(boundary)
    REDUCE(3, 7, boundary)
    __syncthreads();

    //WRITE_SHARED(7)
    //__syncthreads();

    /**/
    w_idx = tid / 2 + ((tid % 2 == 0) ? offset0 : offset1);
    a_shared[w_idx] = ra7;
    b_shared[w_idx] = rb7;
    c_shared[w_idx] = rc7;
    d_shared[w_idx] = rd7;
    __syncthreads();
    //CR Forward

    for(int iter = 0; iter < iter_num; ++iter) {
        int offset2 = offset_table[iter+2];
        val_t aa, bb, cc, dd;
        if(tid < n_even) {
            bool boundary = tid + 1 >= n_odd;
            val_t k1 = - a_shared[tid + offset1] / b_shared[tid + offset0];
            val_t k2 = - c_shared[tid + offset1] / (boundary ? 1 : b_shared[tid + offset0 + 1]);
            aa = a_shared[tid + offset0] * k1;
            bb = b_shared[tid + offset1] + c_shared[tid + offset0] * k1 \
                    + (boundary ? ZERO(val_t) : a_shared[tid + offset0 + 1]) * k2;
            cc = (boundary ? ZERO(val_t) : c_shared[tid + offset0 + 1]) * k2;
            dd = d_shared[tid + offset1] + d_shared[tid + offset0] * k1 \
                    + (boundary ? ZERO(val_t) : d_shared[tid + offset0 + 1]) * k2;
        }
        __syncthreads();
        if(tid < n_even) {
            w_idx = tid/2 + ((tid % 2 == 0) ? offset1 : offset2);
            a_shared[w_idx] = aa;
            b_shared[w_idx] = bb;
            c_shared[w_idx] = cc;
            d_shared[w_idx] = dd;
        }
        __syncthreads();
        n = n_even;
        n_even = n / 2;
        n_odd = n - n_even;
        offset0 = offset1;
        offset1 = offset2;
    }
    if(tid == 0){
        if(n == 2) {
            val_t det = b_shared[offset0] * b_shared[offset1] - c_shared[offset0] * a_shared[offset1];
            val_t x0 = d_shared[offset0] * b_shared[offset1] - c_shared[offset0] * d_shared[offset1];
            val_t x1 = b_shared[offset0] * d_shared[offset1] - d_shared[offset0] * a_shared[offset1];
            d_shared[offset0] = x0 / det;
            d_shared[offset0+1] = x1 / det;
        } else {
            d_shared[offset0] = d_shared[offset0] / b_shared[offset0];
        }
    }
    __syncthreads();

    for(int iter = iter_num - 1; iter >= 0; --iter) {
        val_t dd0, dd1;
        //int offset2 = offset1;
        offset1 = offset0;
        offset0 = offset_table[iter];
        n_even = n;
        n = length_table[iter];
        n_odd = n - n_even;
        
        if(tid < n_odd) {
            dd1 = (tid >= n_even) ? ZERO(val_t) : d_shared[tid+offset1];
            dd0 = (d_shared[tid+offset0] \
                    - a_shared[tid+offset0] * ((tid == 0) ? ZERO(val_t) : d_shared[tid+offset1-1]) \
                    - c_shared[tid+offset0] * dd1) \
                    / b_shared[tid+offset0];
        }
        __syncthreads();
        if(tid < n_odd) {
            d_shared[tid*2+offset0] = dd0;
            if(tid < n_even) {
                d_shared[tid*2+offset0+1] = dd1;
            }
        }
        __syncthreads();
    }
    /**/


    rd7 = d_shared[tid];
    rdboundary = (tid == 0) ? ZERO(val_t) : d_shared[tid-1];

    SUBSTITUDE(boundary, 3, 7)
    SUBSTITUDE(boundary, 1, 3)
    SUBSTITUDE(3, 5, 7)
    SUBSTITUDE(boundary, 0, 1)
    SUBSTITUDE(1, 2, 3)
    SUBSTITUDE(3, 4, 5)
    SUBSTITUDE(5, 6, 7)

    WRITE_VAR(0, 8)
    WRITE_VAR(1, 8)
    WRITE_VAR(2, 8)
    WRITE_VAR(3, 8)
    WRITE_VAR(4, 8)
    WRITE_VAR(5, 8)
    WRITE_VAR(6, 8)
    WRITE_VAR(7, 8)
}

template<typename val_t>
void CR_reg8(val_t *a_gbl, val_t *b_gbl, val_t *c_gbl, val_t *d_gbl, int n_eqt, int n_batch) {
    int offset_table_host[14];
    int length_table_host[13];
    int n, iter, offset, shmem_size;
    int n_even, n_odd, n_last;
    int N8 = n_eqt / 8;
    n_even = N8 / 2;
    n_odd = N8 - n_even;
    n = n_even;
    offset = n_odd;// + padding(n_odd);
    length_table_host[0] = N8;
    offset_table_host[0] = 0;
    offset_table_host[1] = offset;
    for(iter = 0; n > 1; ++iter, n = n / 2) {
        offset = offset + n;
        length_table_host[iter+1] = n;
        offset_table_host[iter+2] = offset;
        n_last = n;
    }
    if(n_last == 3) {
        length_table_host[iter+1] = 0;
        offset_table_host[iter+2] = offset;
        ++iter;
    }
    shmem_size = offset + n;
    set_table(length_table_host, offset_table_host, iter);
    CR_reg8_kernel<val_t><<<n_batch, N8, shmem_size*4*sizeof(val_t)>>>(
        a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch, shmem_size, iter);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("CR_reg8_kernel");
}


template<typename val_t>
__global__ void CR_reg4_kernel(val_t *a_gbl, val_t *b_gbl, val_t *c_gbl, val_t *d_gbl, int n_eqt, int n_batch, int shmem_size, int iter_num)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    val_t *a = a_gbl + bid*n_eqt;
    val_t *b = b_gbl + bid*n_eqt;
    val_t *c = c_gbl + bid*n_eqt;
    val_t *d = d_gbl + bid*n_eqt;
    extern __shared__ double shared[];
    val_t* a_shared = reinterpret_cast<val_t*>(shared);
    val_t* b_shared = (val_t*)&a_shared[shmem_size];
    val_t* c_shared = (val_t*)&b_shared[shmem_size];
    val_t* d_shared = (val_t*)&c_shared[shmem_size];
    int n = n_eqt / 4, n_even = n / 2, n_odd = n - n_even;
    int offset0 = offset_table[0], offset1 = offset_table[1];
    int w_idx;

    val_t k1, k2;
    DEF_VARS(boundary)
    DEF_VARS(0)
    DEF_VARS(1)
    DEF_VARS(2)
    DEF_VARS(3)

    READ_VAR(0, 4)
    READ_VAR(1, 4)
    READ_VAR(2, 4)
    READ_VAR(3, 4)

    WRITE_SHARED(0)
    __syncthreads();
    READ_SHARED(boundary)
    REDUCE(0, 1, 2)
    REDUCE(2, 3, boundary)
    __syncthreads();

    WRITE_SHARED(1)
    __syncthreads();
    READ_SHARED(boundary)
    REDUCE(1, 3, boundary)
    __syncthreads();


    /**/
    w_idx = tid / 2 + ((tid % 2 == 0) ? offset0 : offset1);
    a_shared[w_idx] = ra3;
    b_shared[w_idx] = rb3;
    c_shared[w_idx] = rc3;
    d_shared[w_idx] = rd3;
    __syncthreads();
    //CR Forward

    for(int iter = 0; iter < iter_num; ++iter) {
        int offset2 = offset_table[iter+2];
        val_t aa, bb, cc, dd;
        if(tid < n_even) {
            bool boundary = tid + 1 >= n_odd;
            val_t k1 = -a_shared[tid + offset1] / b_shared[tid + offset0];
            val_t k2 = -c_shared[tid + offset1] / (boundary ? 1 : b_shared[tid + offset0 + 1]);
            aa = a_shared[tid + offset0] * k1;
            bb = b_shared[tid + offset1] + c_shared[tid + offset0] * k1 \
                    + (boundary ? ZERO(val_t) : a_shared[tid + offset0 + 1]) * k2;
            cc = (boundary ? ZERO(val_t) : c_shared[tid + offset0 + 1]) * k2;
            dd = d_shared[tid + offset1] + d_shared[tid + offset0] * k1 \
                    + (boundary ? ZERO(val_t) : d_shared[tid + offset0 + 1]) * k2;
        }
        __syncthreads();
        if(tid < n_even) {
            w_idx = tid/2 + ((tid % 2 == 0) ? offset1 : offset2);
            a_shared[w_idx] = aa;
            b_shared[w_idx] = bb;
            c_shared[w_idx] = cc;
            d_shared[w_idx] = dd;
        }
        __syncthreads();
        n = n_even;
        n_even = n / 2;
        n_odd = n - n_even;
        offset0 = offset1;
        offset1 = offset2;
    }
    if(tid == 0){
        if(n == 2) {
            val_t det = b_shared[offset0] * b_shared[offset1] - c_shared[offset0] * a_shared[offset1];
            val_t x0 = d_shared[offset0] * b_shared[offset1] - c_shared[offset0] * d_shared[offset1];
            val_t x1 = b_shared[offset0] * d_shared[offset1] - d_shared[offset0] * a_shared[offset1];
            d_shared[offset0] = x0 / det;
            d_shared[offset0+1] = x1 / det;
        } else {
            d_shared[offset0] = d_shared[offset0] / b_shared[offset0];
        }
    }
    __syncthreads();

    for(int iter = iter_num - 1; iter >= 0; --iter) {
        val_t dd0, dd1;
        //int offset2 = offset1;
        offset1 = offset0;
        offset0 = offset_table[iter];
        n_even = n;
        n = length_table[iter];
        n_odd = n - n_even;
        
        if(tid < n_odd) {
            dd1 = (tid >= n_even) ? ZERO(val_t) : d_shared[tid+offset1];
            dd0 = (d_shared[tid+offset0] \
                    - a_shared[tid+offset0] * ((tid == 0) ? ZERO(val_t) : d_shared[tid+offset1-1]) \
                    - c_shared[tid+offset0] * dd1) \
                    / b_shared[tid+offset0];
        }
        __syncthreads();
        if(tid < n_odd) {
            d_shared[tid*2+offset0] = dd0;
            if(tid < n_even) {
                d_shared[tid*2+offset0+1] = dd1;
            }
        }
        __syncthreads();
    }
    /**/


    rd3 = d_shared[tid];
    rdboundary = (tid == 0) ? ZERO(val_t) : d_shared[tid-1];

    SUBSTITUDE(boundary, 1, 3)
    SUBSTITUDE(boundary, 0, 1)
    SUBSTITUDE(1, 2, 3)

    WRITE_VAR(0, 4)
    WRITE_VAR(1, 4)
    WRITE_VAR(2, 4)
    WRITE_VAR(3, 4)
}

template<typename val_t>
void CR_reg4(val_t *a_gbl, val_t *b_gbl, val_t *c_gbl, val_t *d_gbl, int n_eqt, int n_batch) {
    int offset_table_host[14];
    int length_table_host[13];
    int n, iter, offset, shmem_size;
    int n_even, n_odd, n_last;
    int N4 = n_eqt / 4;
    n_even = N4 / 2;
    n_odd = N4 - n_even;
    n = n_even;
    offset = n_odd;// + padding(n_odd);
    length_table_host[0] = N4;
    offset_table_host[0] = 0;
    offset_table_host[1] = offset;
    for(iter = 0; n > 1; ++iter, n = n / 2) {
        offset = offset + n;
        length_table_host[iter+1] = n;
        offset_table_host[iter+2] = offset;
        n_last = n;
    }
    if(n_last == 3) {
        length_table_host[iter+1] = 0;
        offset_table_host[iter+2] = offset;
        ++iter;
    }
    shmem_size = offset + n;
    set_table(length_table_host, offset_table_host, iter);
    CR_reg4_kernel<val_t><<<n_batch, N4, shmem_size*4*sizeof(val_t)>>>(
        a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch, shmem_size, iter);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("CR_reg4_kernel");
}

template<typename val_t>
__global__ void CR_reg16_kernel(val_t *a_gbl, val_t *b_gbl, val_t *c_gbl, val_t *d_gbl, int n_eqt, int n_batch, int shmem_size, int iter_num)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    val_t *a = a_gbl + bid*n_eqt;
    val_t *b = b_gbl + bid*n_eqt;
    val_t *c = c_gbl + bid*n_eqt;
    val_t *d = d_gbl + bid*n_eqt;
    extern __shared__ double shared[];
    val_t* a_shared = reinterpret_cast<val_t*>(shared);
    val_t* b_shared = (val_t*)&a_shared[shmem_size];
    val_t* c_shared = (val_t*)&b_shared[shmem_size];
    val_t* d_shared = (val_t*)&c_shared[shmem_size];
    int n = n_eqt / 16, n_even = n / 2, n_odd = n - n_even;
    int offset0 = offset_table[0], offset1 = offset_table[1];
    int w_idx;

    val_t k1, k2;
    DEF_VARS(boundary)
    DEF_VARS(0)
    DEF_VARS(1)
    DEF_VARS(2)
    DEF_VARS(3)
    DEF_VARS(4)
    DEF_VARS(5)
    DEF_VARS(6)
    DEF_VARS(7)
    DEF_VARS(8)
    DEF_VARS(9)
    DEF_VARS(10)
    DEF_VARS(11)
    DEF_VARS(12)
    DEF_VARS(13)
    DEF_VARS(14)
    DEF_VARS(15)

    READ_VAR(0, 16)
    READ_VAR(1, 16)
    READ_VAR(2, 16)
    READ_VAR(3, 16)
    READ_VAR(4, 16)
    READ_VAR(5, 16)
    READ_VAR(6, 16)
    READ_VAR(7, 16)
    READ_VAR(8, 16)
    READ_VAR(9, 16)
    READ_VAR(10, 16)
    READ_VAR(11, 16)
    READ_VAR(12, 16)
    READ_VAR(13, 16)
    READ_VAR(14, 16)
    READ_VAR(15, 16)

    WRITE_SHARED(0)
    __syncthreads();
    READ_SHARED(boundary)
    REDUCE(0, 1, 2)
    REDUCE(2, 3, 4)
    REDUCE(4, 5, 6)
    REDUCE(6, 7, 8)
    REDUCE(8, 9, 10)
    REDUCE(10,11,12)
    REDUCE(12,13,14)
    REDUCE(14,15,boundary)
    __syncthreads();

    WRITE_SHARED(1)
    __syncthreads();
    READ_SHARED(boundary)
    REDUCE(1, 3, 5)
    REDUCE(5, 7, 9)
    REDUCE(9, 11,13)
    REDUCE(13,15,boundary)
    __syncthreads();

    WRITE_SHARED(3)
    __syncthreads();
    READ_SHARED(boundary)
    REDUCE(3, 7, 11)
    REDUCE(11, 15, boundary)
    __syncthreads();

    WRITE_SHARED(7)
    __syncthreads();
    READ_SHARED(boundary)
    REDUCE(7, 15, boundary)
    __syncthreads();

    /**/
    w_idx = tid / 2 + ((tid % 2 == 0) ? offset0 : offset1);
    a_shared[w_idx] = ra15;
    b_shared[w_idx] = rb15;
    c_shared[w_idx] = rc15;
    d_shared[w_idx] = rd15;
    __syncthreads();
    //CR Forward

    for(int iter = 0; iter < iter_num; ++iter) {
        int offset2 = offset_table[iter+2];
        val_t aa, bb, cc, dd;
        if(tid < n_even) {
            bool boundary = tid + 1 >= n_odd;
            val_t k1 = - a_shared[tid + offset1] / b_shared[tid + offset0];
            val_t k2 = - c_shared[tid + offset1] / (boundary ? 1 : b_shared[tid + offset0 + 1]);
            aa = a_shared[tid + offset0] * k1;
            bb = b_shared[tid + offset1] + c_shared[tid + offset0] * k1 \
                    + (boundary ? ZERO(val_t) : a_shared[tid + offset0 + 1]) * k2;
            cc = (boundary ? ZERO(val_t) : c_shared[tid + offset0 + 1]) * k2;
            dd = d_shared[tid + offset1] + d_shared[tid + offset0] * k1 \
                    + (boundary ? ZERO(val_t) : d_shared[tid + offset0 + 1]) * k2;
        }
        __syncthreads();
        if(tid < n_even) {
            w_idx = tid/2 + ((tid % 2 == 0) ? offset1 : offset2);
            a_shared[w_idx] = aa;
            b_shared[w_idx] = bb;
            c_shared[w_idx] = cc;
            d_shared[w_idx] = dd;
        }
        __syncthreads();
        n = n_even;
        n_even = n / 2;
        n_odd = n - n_even;
        offset0 = offset1;
        offset1 = offset2;
    }
    if(tid == 0){
        if(n == 2) {
            val_t det = b_shared[offset0] * b_shared[offset1] - c_shared[offset0] * a_shared[offset1];
            val_t x0 = d_shared[offset0] * b_shared[offset1] - c_shared[offset0] * d_shared[offset1];
            val_t x1 = b_shared[offset0] * d_shared[offset1] - d_shared[offset0] * a_shared[offset1];
            d_shared[offset0] = x0 / det;
            d_shared[offset0+1] = x1 / det;
        } else {
            d_shared[offset0] = d_shared[offset0] / b_shared[offset0];
        }
    }
    __syncthreads();

    for(int iter = iter_num - 1; iter >= 0; --iter) {
        val_t dd0, dd1;
        //int offset2 = offset1;
        offset1 = offset0;
        offset0 = offset_table[iter];
        n_even = n;
        n = length_table[iter];
        n_odd = n - n_even;
        
        if(tid < n_odd) {
            dd1 = (tid >= n_even) ? ZERO(val_t) : d_shared[tid+offset1];
            dd0 = (d_shared[tid+offset0] \
                    - a_shared[tid+offset0] * ((tid == 0) ? ZERO(val_t) : d_shared[tid+offset1-1]) \
                    - c_shared[tid+offset0] * dd1) \
                    / b_shared[tid+offset0];
        }
        __syncthreads();
        if(tid < n_odd) {
            d_shared[tid*2+offset0] = dd0;
            if(tid < n_even) {
                d_shared[tid*2+offset0+1] = dd1;
            }
        }
        __syncthreads();
    }
    /**/


    rd15 = d_shared[tid];
    rdboundary = (tid == 0) ? ZERO(val_t) : d_shared[tid-1];

    SUBSTITUDE(boundary, 7, 15)
    SUBSTITUDE(boundary, 3, 7)
    SUBSTITUDE(7,        11,15)
    SUBSTITUDE(boundary, 1, 3)
    SUBSTITUDE(3, 5, 7)
    SUBSTITUDE(7, 9, 11)
    SUBSTITUDE(11, 13, 15)
    SUBSTITUDE(boundary, 0, 1)
    SUBSTITUDE(1, 2, 3)
    SUBSTITUDE(3, 4, 5)
    SUBSTITUDE(5, 6, 7)
    SUBSTITUDE(7, 8, 9)
    SUBSTITUDE(9, 10, 11)
    SUBSTITUDE(11, 12, 13)
    SUBSTITUDE(13, 14, 15)
    
    WRITE_VAR(0, 16)
    WRITE_VAR(1, 16)
    WRITE_VAR(2, 16)
    WRITE_VAR(3, 16)
    WRITE_VAR(4, 16)
    WRITE_VAR(5, 16)
    WRITE_VAR(6, 16)
    WRITE_VAR(7, 16)
    WRITE_VAR(8, 16)
    WRITE_VAR(9, 16)
    WRITE_VAR(10, 16)
    WRITE_VAR(11, 16)
    WRITE_VAR(12, 16)
    WRITE_VAR(13, 16)
    WRITE_VAR(14, 16)
    WRITE_VAR(15, 16)
}

template<typename val_t>
void CR_reg16(val_t *a_gbl, val_t *b_gbl, val_t *c_gbl, val_t *d_gbl, int n_eqt, int n_batch) {
    int offset_table_host[14];
    int length_table_host[13];
    int n, iter, offset, shmem_size;
    int n_even, n_odd, n_last;
    int N16 = n_eqt / 16;
    n_even = N16 / 2;
    n_odd = N16 - n_even;
    n = n_even;
    offset = n_odd;// + padding(n_odd);
    length_table_host[0] = N16;
    offset_table_host[0] = 0;
    offset_table_host[1] = offset;
    for(iter = 0; n > 1; ++iter, n = n / 2) {
        offset = offset + n;
        length_table_host[iter+1] = n;
        offset_table_host[iter+2] = offset;
        n_last = n;
    }
    if(n_last == 3) {
        length_table_host[iter+1] = 0;
        offset_table_host[iter+2] = offset;
        ++iter;
    }
    shmem_size = offset + n;
    set_table(length_table_host, offset_table_host, iter);
    CR_reg16_kernel<val_t><<<n_batch, N16, shmem_size*4*sizeof(val_t)>>>(
        a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch, shmem_size, iter);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("CR_reg16_kernel");
}



class TridiagonalConfig_1 {
public:
    TridiagonalConfig_1() {
        cudaFuncSetSharedMemConfig(CR_shmem_kernel<float>,   cudaSharedMemBankSizeFourByte);
        cudaFuncSetSharedMemConfig(CR_reg4_kernel<float>,   cudaSharedMemBankSizeFourByte);
        cudaFuncSetSharedMemConfig(CR_reg8_kernel<float>,   cudaSharedMemBankSizeFourByte);
        cudaFuncSetSharedMemConfig(CR_reg16_kernel<float>,   cudaSharedMemBankSizeFourByte);
        cudaFuncSetSharedMemConfig(CR_shmem_kernel<double>,  cudaSharedMemBankSizeEightByte);
        cudaFuncSetSharedMemConfig(CR_reg4_kernel<double>,  cudaSharedMemBankSizeEightByte);
        cudaFuncSetSharedMemConfig(CR_reg8_kernel<double>,  cudaSharedMemBankSizeEightByte);
        cudaFuncSetSharedMemConfig(CR_reg16_kernel<double>,  cudaSharedMemBankSizeEightByte);
        cudaFuncSetCacheConfig(CR_shmem_kernel<float>, cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(CR_reg4_kernel<float>,  cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(CR_reg8_kernel<float>,  cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(CR_reg16_kernel<float>, cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(CR_shmem_kernel<double>, cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(CR_reg4_kernel<double>,  cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(CR_reg8_kernel<double>,  cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(CR_reg16_kernel<double>, cudaFuncCachePreferShared);
    }
};

static const TridiagonalConfig_1 config;

void CR_reg4_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch) {
    CR_reg4(a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
}

void CR_reg8_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch) {
    CR_reg8(a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
}

void CR_reg16_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch) {
    CR_reg16(a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
}

void CR_reg4_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch) {
    CR_reg4(a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
}

void CR_reg8_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch) {
    CR_reg8(a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
}

void CR_reg16_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch) {
    CR_reg16(a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
}

