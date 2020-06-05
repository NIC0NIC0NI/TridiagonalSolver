#include "tridiagonal.h"
#include "intrin.h"

#define WARP_SIZE 32
#define ZERO(type) static_cast<type>(0)  /* otherwise nvcc use cvt instructions */
#define ONE(type)  static_cast<type>(1)  /* otherwise nvcc use cvt instructions */

#define shfl(a, lane) __shfl_sync(shuffle_mask, a, lane, warp_size)
#define shfl_xor(a, xor_mask) __shfl_xor_sync(shuffle_mask, a, xor_mask, warp_size)
#define concatenate(a, b, tid, offset) __shfl_sync(shuffle_mask, (tid < offset) ? a : b, tid + offset, warp_size)

/*  According to Nvidia PTX document, fnmadd (d = c - a * b) is not supported,
 *  therefore, k should be negated in advance
 */


template<typename val_t, int BLOCK_FUSION>
__global__ void WM5_shmem_kernel(val_t *a_gbl, val_t *b_gbl, val_t *c_gbl, val_t *d_gbl, int n_eqt, int n_batch) {
    const int tid = threadIdx.x & 31;
    const int bid = (blockIdx.x * (BLOCK_FUSION * WARP_SIZE) + threadIdx.x) >> 5;
    if(bid >= n_batch) { return; }
    __shared__ val_t aa_shared[BLOCK_FUSION][64];
    __shared__ val_t bb_shared[BLOCK_FUSION][64];
    __shared__ val_t cc_shared[BLOCK_FUSION][64];
    __shared__ val_t dd_shared[BLOCK_FUSION][64];
    val_t *a_shared = aa_shared[threadIdx.x >> 5];
    val_t *b_shared = bb_shared[threadIdx.x >> 5];
    val_t *c_shared = cc_shared[threadIdx.x >> 5];
    val_t *d_shared = dd_shared[threadIdx.x >> 5];
    val_t *a_mem = a_gbl + bid*n_eqt;
    val_t *b_mem = b_gbl + bid*n_eqt;
    val_t *c_mem = c_gbl + bid*n_eqt;
    val_t *d_mem = d_gbl + bid*n_eqt;
    
    val_t last_c = ZERO(val_t), last_d = ZERO(val_t);
    for(int i = 0; i < n_eqt; i += 2 * WARP_SIZE) {
        auto a2 = load_2(a_mem + i + tid * 2);
        auto b2 = load_2(b_mem + i + tid * 2);
        auto c2 = load_2(c_mem + i + tid * 2);
        auto d2 = load_2(d_mem + i + tid * 2);
        a_shared[tid] = a2.x;
        b_shared[tid] = b2.x;
        c_shared[tid] = c2.x;
        d_shared[tid] = d2.x;
        a_shared[tid|32] = a2.y;
        b_shared[tid|32] = b2.y;
        c_shared[tid|32] = c2.y;
        d_shared[tid|32] = d2.y;

        int ids, idd, idll, idhf;
        val_t k1, k2, k3, e, at, bt, ct, dt;
        /* ROUND 1 */
        k1         =   div(a_shared[tid|32],     - b_shared[tid]);
        b_shared[tid|32] = b_shared[tid|32] + k1 * c_shared[tid];
        a_shared[tid|32] =                    k1 * a_shared[tid];
        d_shared[tid|32] = d_shared[tid|32] + k1 * d_shared[tid];
        k2         =   div(c_shared[tid],        - b_shared[tid|32]);
        a_shared[tid]    = a_shared[tid]    + k2 * a_shared[tid|32];
        c_shared[tid]    =                    k2 * c_shared[tid|32];
        d_shared[tid]    = d_shared[tid]    + k2 * d_shared[tid|32];
        __syncwarp();
        ids = (tid ^ 1) | ((tid & 1) << 5);
        idd = tid | (((tid & 1) << 5) ^ 32);
        at = a_shared[ids], bt = b_shared[ids], ct = c_shared[ids], dt = d_shared[ids];
        __syncwarp();
        a_shared[idd] = at, b_shared[idd] = bt, c_shared[idd] = ct, d_shared[idd] = dt;
        __syncwarp();

        /* ROUND 2 */
        idll = tid | 1;
        idhf = (tid & (~1)) | 32;
        k1         =       div(a_shared[tid|32], - b_shared[idll]);
        e = (((tid & 1) == 0) ? b_shared[tid|32] : ZERO(val_t)) + k1 * c_shared[idll];
        a_shared[tid|32] =                    k1 * a_shared[idll];
        d_shared[tid|32] = d_shared[tid|32] + k1 * d_shared[idll];
        if((tid & 1) == 0) b_shared[tid|32] = e;
        __syncwarp();
        k2         =   div(c_shared[tid],        - b_shared[idhf]);
        k3 = maskz_div((tid&1) == 0, e, - b_shared[idhf]);
        a_shared[tid]    = a_shared[tid]    + k2 * a_shared[idhf];
        c_shared[tid]    =                    k2 * c_shared[idhf];
        d_shared[tid]    = d_shared[tid]    + k2 * d_shared[idhf];
        a_shared[tid|32] = a_shared[tid|32] + k3 * a_shared[idhf];
        c_shared[tid|32] = c_shared[tid|32] + k3 * c_shared[idhf];
        d_shared[tid|32] = d_shared[tid|32] + k3 * d_shared[idhf];
        __syncwarp();
        ids = (tid ^ 2) | ((tid & 2) << 4);
        idd = tid | (((tid & 2) << 4) ^ 32);
        at = a_shared[ids], bt = b_shared[ids], ct = c_shared[ids], dt = d_shared[ids];
        __syncwarp();
        a_shared[idd] = at, b_shared[idd] = bt, c_shared[idd] = ct, d_shared[idd] = dt;
        __syncwarp();
  
        /* ROUND 3 */
        idll = tid | 3;
        idhf = (tid & (~3)) | 32;
        k1 = div(a_shared[tid|32], - b_shared[idll]);
        e = (((tid & 3) == 0) ? b_shared[tid|32] : ZERO(val_t)) + k1 * c_shared[idll];
        a_shared[tid|32] =                    k1 * a_shared[idll];
        d_shared[tid|32] = d_shared[tid|32] + k1 * d_shared[idll];
        if((tid & 3) == 0) b_shared[tid|32] = e;
        __syncwarp();
        k2         =   div(c_shared[tid],        - b_shared[idhf]);
        k3 = maskz_div((tid&3) == 0, e, - b_shared[idhf]);
        a_shared[tid]    = a_shared[tid]    + k2 * a_shared[idhf];
        c_shared[tid]    =                    k2 * c_shared[idhf];
        d_shared[tid]    = d_shared[tid]    + k2 * d_shared[idhf];
        a_shared[tid|32] = a_shared[tid|32] + k3 * a_shared[idhf];
        c_shared[tid|32] = c_shared[tid|32] + k3 * c_shared[idhf];
        d_shared[tid|32] = d_shared[tid|32] + k3 * d_shared[idhf];
        __syncwarp();
        ids = (tid ^ 4) | ((tid & 4) << 3);
        idd = tid | (((tid & 4) << 3) ^ 32);
        at = a_shared[ids], bt = b_shared[ids], ct = c_shared[ids], dt = d_shared[ids];
        __syncwarp();
        a_shared[idd] = at, b_shared[idd] = bt, c_shared[idd] = ct, d_shared[idd] = dt;
        __syncwarp();
        
        /* ROUND 4 */
        idll = tid | 7;
        idhf = (tid & (~7)) | 32;
        k1 = div(a_shared[tid|32], - b_shared[idll]);
        e = (((tid & 7) == 0) ? b_shared[tid|32] : ZERO(val_t)) + k1 * c_shared[idll];
        a_shared[tid|32] =                    k1 * a_shared[idll];
        d_shared[tid|32] = d_shared[tid|32] + k1 * d_shared[idll];
        if((tid & 7) == 0) b_shared[tid|32] = e;
        __syncwarp();
        k2         = div(c_shared[tid],          - b_shared[idhf]);
        k3 = maskz_div((tid&7) == 0, e, - b_shared[idhf]);
        a_shared[tid]    = a_shared[tid]    + k2 * a_shared[idhf];
        c_shared[tid]    =                    k2 * c_shared[idhf];
        d_shared[tid]    = d_shared[tid]    + k2 * d_shared[idhf];
        a_shared[tid|32] = a_shared[tid|32] + k3 * a_shared[idhf];
        c_shared[tid|32] = c_shared[tid|32] + k3 * c_shared[idhf];
        d_shared[tid|32] = d_shared[tid|32] + k3 * d_shared[idhf];
        __syncwarp();
        ids = (tid ^ 8) | ((tid & 8) << 2);
        idd = tid | (((tid & 8) << 2) ^ 32);
        at = a_shared[ids], bt = b_shared[ids], ct = c_shared[ids], dt = d_shared[ids];
        __syncwarp();
        a_shared[idd] = at, b_shared[idd] = bt, c_shared[idd] = ct, d_shared[idd] = dt;
        __syncwarp();
  
        /* ROUND 5 */
        idll = tid | 15;
        idhf = (tid & (~15)) | 32;
        k1           = div(a_shared[tid|32],     - b_shared[idll]);
        e = (((tid & 15) == 0) ? b_shared[tid|32] : ZERO(val_t)) + k1 * c_shared[idll];
        a_shared[tid|32] =                    k1 * a_shared[idll];
        d_shared[tid|32] = d_shared[tid|32] + k1 * d_shared[idll];
        if((tid & 15) == 0) b_shared[tid|32] = e;
        __syncwarp();
        k2         =   div(c_shared[tid],        - b_shared[idhf]);
        k3 = maskz_div((tid&15) == 0, e, - b_shared[idhf]);
        a_shared[tid]    = a_shared[tid]    + k2 * a_shared[idhf];
        c_shared[tid]    =                    k2 * c_shared[idhf];
        d_shared[tid]    = d_shared[tid]    + k2 * d_shared[idhf];
        a_shared[tid|32] = a_shared[tid|32] + k3 * a_shared[idhf];
        c_shared[tid|32] = c_shared[tid|32] + k3 * c_shared[idhf];
        d_shared[tid|32] = d_shared[tid|32] + k3 * d_shared[idhf];
        __syncwarp();
        ids = (tid ^ 16) | ((tid & 16) << 1);
        idd = tid | (((tid & 16) << 1) ^ 32);
        at = a_shared[ids], bt = b_shared[ids], ct = c_shared[ids], dt = d_shared[ids];
        __syncwarp();
        a_shared[idd] = at, b_shared[idd] = bt, c_shared[idd] = ct, d_shared[idd] = dt;
        __syncwarp();
  
        k1 = - a_shared[tid];
        e = ((tid == 0) ? b_shared[tid] : ZERO(val_t)) + k1 * last_c;
        d_shared[tid] =   d_shared[tid]        + k1 * last_d;
        if(tid == 0)  b_shared[tid] = e;
        __syncwarp();
        k3  = maskz_div(tid == 0, e, - b_shared[0]);
        c_shared[tid] = div(c_shared[tid] + k3 * c_shared[0], b_shared[tid]);
        d_shared[tid] = div(d_shared[tid] + k3 * d_shared[0], b_shared[tid]);
        c_mem[tid+i] = c_shared[tid];
        d_mem[tid+i] = d_shared[tid];
        __syncwarp();

        k1 = - a_shared[tid|32];
        e = ((tid == 0) ?  b_shared[tid|32] : ZERO(val_t)) + k1 * c_shared[31];
        d_shared[tid|32] = d_shared[tid|32]        + k1 * d_shared[31];
        if(tid == 0)  b_shared[tid|32] = e;
        __syncwarp();
        k3 = maskz_div(tid == 0, e, - b_shared[32]);
        c_shared[tid|32] = div(c_shared[tid|32] + k3 * c_shared[32], b_shared[tid|32]);
        d_shared[tid|32] = div(d_shared[tid|32] + k3 * d_shared[32], b_shared[tid|32]);
        c_mem[i + (tid|32)] = c_shared[tid|32];
        d_mem[i + (tid|32)] = d_shared[tid|32];
        __syncwarp();
        last_c = c_shared[63];
        last_d = d_shared[63];
        __syncwarp();
    }
    val_t last_x = d_mem[n_eqt - WARP_SIZE];
    for(int i = n_eqt-2*WARP_SIZE; i >= 0; i -= WARP_SIZE) {
        val_t x = d_mem[i+tid] - last_x * c_mem[i+tid];
        d_shared[tid] = x;
        d_mem[i+tid] = x;
        __syncwarp();
        last_x = d_shared[0];
        __syncwarp();
    }
}

template<typename val_t, int BLOCK_FUSION>
__global__ void WM4_shmem_kernel(val_t *a_gbl, val_t *b_gbl, val_t *c_gbl, val_t *d_gbl, int n_eqt, int n_batch) {
    const int tid = threadIdx.x & 31;
    const int bid = (blockIdx.x * (BLOCK_FUSION * WARP_SIZE) + threadIdx.x) >> 5;
    if(bid >= n_batch) { return; }
    __shared__ val_t aa_shared[BLOCK_FUSION][64];
    __shared__ val_t bb_shared[BLOCK_FUSION][64];
    __shared__ val_t cc_shared[BLOCK_FUSION][64];
    __shared__ val_t dd_shared[BLOCK_FUSION][64];
    val_t *a_shared = aa_shared[threadIdx.x >> 5];
    val_t *b_shared = bb_shared[threadIdx.x >> 5];
    val_t *c_shared = cc_shared[threadIdx.x >> 5];
    val_t *d_shared = dd_shared[threadIdx.x >> 5];
    val_t *a_mem = a_gbl + bid*n_eqt;
    val_t *b_mem = b_gbl + bid*n_eqt;
    val_t *c_mem = c_gbl + bid*n_eqt;
    val_t *d_mem = d_gbl + bid*n_eqt;
    
    val_t last_c = ZERO(val_t), last_d = ZERO(val_t);
    for(int i = 0; i < n_eqt; i += 2 * WARP_SIZE) {
        auto a2 = load_2(a_mem + i + tid * 2);
        auto b2 = load_2(b_mem + i + tid * 2);
        auto c2 = load_2(c_mem + i + tid * 2);
        auto d2 = load_2(d_mem + i + tid * 2);
        a_shared[tid] = a2.x;
        b_shared[tid] = b2.x;
        c_shared[tid] = c2.x;
        d_shared[tid] = d2.x;
        a_shared[tid|32] = a2.y;
        b_shared[tid|32] = b2.y;
        c_shared[tid|32] = c2.y;
        d_shared[tid|32] = d2.y;

        int ids, idd, idll, idhf;
        val_t k1, k2, k3, e, at, bt, ct, dt;
        /* ROUND 1 */
        k1         =   div(a_shared[tid|32],     - b_shared[tid]);
        b_shared[tid|32] = b_shared[tid|32] + k1 * c_shared[tid];
        a_shared[tid|32] =                    k1 * a_shared[tid];
        d_shared[tid|32] = d_shared[tid|32] + k1 * d_shared[tid];
        k2         =   div(c_shared[tid],        - b_shared[tid|32]);
        a_shared[tid]    = a_shared[tid]    + k2 * a_shared[tid|32];
        c_shared[tid]    =                    k2 * c_shared[tid|32];
        d_shared[tid]    = d_shared[tid]    + k2 * d_shared[tid|32];
        __syncwarp();
        ids = (tid ^ 1) | ((tid & 1) << 5);
        idd = tid | (((tid & 1) << 5) ^ 32);
        at = a_shared[ids], bt = b_shared[ids], ct = c_shared[ids], dt = d_shared[ids];
        __syncwarp();
        a_shared[idd] = at, b_shared[idd] = bt, c_shared[idd] = ct, d_shared[idd] = dt;
        __syncwarp();

        /* ROUND 2 */
        idll = tid | 1;
        idhf = (tid & (~1)) | 32;
        k1               = div(a_shared[tid|32], - b_shared[idll]);
        e = (((tid & 1) == 0) ? b_shared[tid|32] : ZERO(val_t)) + k1 * c_shared[idll];
        a_shared[tid|32] =                    k1 * a_shared[idll];
        d_shared[tid|32] = d_shared[tid|32] + k1 * d_shared[idll];
        if((tid & 1) == 0) b_shared[tid|32] = e;
        __syncwarp();
        k2         = div(c_shared[tid],          - b_shared[idhf]);
        k3 = maskz_div((tid&1) == 0, e, - b_shared[idhf]);
        a_shared[tid]    = a_shared[tid]    + k2 * a_shared[idhf];
        c_shared[tid]    =                    k2 * c_shared[idhf];
        d_shared[tid]    = d_shared[tid]    + k2 * d_shared[idhf];
        a_shared[tid|32] = a_shared[tid|32] + k3 * a_shared[idhf];
        c_shared[tid|32] = c_shared[tid|32] + k3 * c_shared[idhf];
        d_shared[tid|32] = d_shared[tid|32] + k3 * d_shared[idhf];
        __syncwarp();
        ids = (tid ^ 2) | ((tid & 2) << 4);
        idd = tid | (((tid & 2) << 4) ^ 32);
        at = a_shared[ids], bt = b_shared[ids], ct = c_shared[ids], dt = d_shared[ids];
        __syncwarp();
        a_shared[idd] = at, b_shared[idd] = bt, c_shared[idd] = ct, d_shared[idd] = dt;
        __syncwarp();
  
        /* ROUND 3 */
        idll = tid | 3;
        idhf = (tid & (~3)) | 32;
        k1 =               div(a_shared[tid|32], - b_shared[idll]);
        e = (((tid & 3) == 0) ? b_shared[tid|32] : ZERO(val_t)) + k1 * c_shared[idll];
        a_shared[tid|32] =                    k1 * a_shared[idll];
        d_shared[tid|32] = d_shared[tid|32] + k1 * d_shared[idll];
        if((tid & 3) == 0) b_shared[tid|32] = e;
        __syncwarp();
        k2         = div(c_shared[tid],          - b_shared[idhf]);
        k3 = maskz_div((tid&3) == 0, e, - b_shared[idhf]);
        a_shared[tid]    = a_shared[tid]    + k2 * a_shared[idhf];
        c_shared[tid]    =                    k2 * c_shared[idhf];
        d_shared[tid]    = d_shared[tid]    + k2 * d_shared[idhf];
        a_shared[tid|32] = a_shared[tid|32] + k3 * a_shared[idhf];
        c_shared[tid|32] = c_shared[tid|32] + k3 * c_shared[idhf];
        d_shared[tid|32] = d_shared[tid|32] + k3 * d_shared[idhf];
        __syncwarp();
        ids = (tid ^ 4) | ((tid & 4) << 3);
        idd = tid | (((tid & 4) << 3) ^ 32);
        at = a_shared[ids], bt = b_shared[ids], ct = c_shared[ids], dt = d_shared[ids];
        __syncwarp();
        a_shared[idd] = at, b_shared[idd] = bt, c_shared[idd] = ct, d_shared[idd] = dt;
        __syncwarp();
        
        /* ROUND 4 */
        idll = tid | 7;
        idhf = (tid & (~7)) | 32;
        k1 =               div(a_shared[tid|32], - b_shared[idll]);
        e = (((tid & 7) == 0) ? b_shared[tid|32] : ZERO(val_t)) + k1 * c_shared[idll];
        a_shared[tid|32] =                    k1 * a_shared[idll];
        d_shared[tid|32] = d_shared[tid|32] + k1 * d_shared[idll];
        if((tid & 7) == 0) b_shared[tid|32] = e;
        __syncwarp();
        k2         = div(c_shared[tid],          - b_shared[idhf]);
        k3 = maskz_div((tid&7) == 0, e, - b_shared[idhf]);
        a_shared[tid]    = a_shared[tid]    + k2 * a_shared[idhf];
        c_shared[tid]    =                    k2 * c_shared[idhf];
        d_shared[tid]    = d_shared[tid]    + k2 * d_shared[idhf];
        a_shared[tid|32] = a_shared[tid|32] + k3 * a_shared[idhf];
        c_shared[tid|32] = c_shared[tid|32] + k3 * c_shared[idhf];
        d_shared[tid|32] = d_shared[tid|32] + k3 * d_shared[idhf];
        __syncwarp();
        ids = (tid ^ 8) | ((tid & 8) << 2);
        idd = tid | (((tid & 8) << 2) ^ 32);
        at = a_shared[ids], bt = b_shared[ids], ct = c_shared[ids], dt = d_shared[ids];
        __syncwarp();
        a_shared[idd] = at, b_shared[idd] = bt, c_shared[idd] = ct, d_shared[idd] = dt;
        __syncwarp();
  
        /* ROUND 5, do nothing */
        __syncwarp();
        ids = (tid ^ 16) | ((tid & 16) << 1);
        idd = tid | (((tid & 16) << 1) ^ 32);
        at = a_shared[ids], bt = b_shared[ids], ct = c_shared[ids], dt = d_shared[ids];
        __syncwarp();
        a_shared[idd] = at, b_shared[idd] = bt, c_shared[idd] = ct, d_shared[idd] = dt;
        __syncwarp();
  
        if(tid < 16) {
            k1 = - a_shared[tid];
            e = ((tid == 0) ? b_shared[0] : ZERO(val_t)) + k1 * last_c;
            d_shared[tid] =   d_shared[tid]              + k1 * last_d;
            if(tid == 0)  b_shared[0] = e;
        }
        __syncwarp();
        if(tid < 16 && tid > 0) {
            k3 = div(e, - b_shared[0]);
            c_shared[tid] = c_shared[tid] + k3 * c_shared[0];
            d_shared[tid] = d_shared[tid] + k3 * d_shared[0];
        }
        __syncwarp();
        if(tid >= 16) {
            k1 = div(a_shared[tid], - b_shared[15]);
            e = ((tid == 16) ? b_shared[16] : ZERO(val_t)) + k1 * c_shared[15];
            d_shared[tid] =    d_shared[tid]               + k1 * d_shared[15];
            if(tid == 16)  b_shared[16] = e;
        }
        __syncwarp();
        if(tid > 16) {
            k3 = div(e, - b_shared[16]);
            c_shared[tid] = c_shared[tid] + k3 * c_shared[16];
            d_shared[tid] = d_shared[tid] + k3 * d_shared[16];
        }
        c_shared[tid] = div(c_shared[tid], b_shared[tid]);
        d_shared[tid] = div(d_shared[tid], b_shared[tid]);
        c_mem[i + tid] = c_shared[tid];
        d_mem[i + tid] = d_shared[tid];
        __syncwarp();

        
        if(tid < 16) {
            k1 = - a_shared[tid | 32];
            e = ((tid == 0) ?    b_shared[32] : ZERO(val_t)) + k1 * c_shared[31];
            d_shared[tid | 32] = d_shared[tid | 32]  + k1 * d_shared[31];
            if(tid == 0)  b_shared[32] = e;
        }
        __syncwarp();
        if(tid < 16 && tid > 0) {
            k3 = div(e, - b_shared[32]);
            c_shared[tid | 32] = c_shared[tid | 32] + k3 * c_shared[32];
            d_shared[tid | 32] = d_shared[tid | 32] + k3 * d_shared[32];
        }
        __syncwarp();
        if(tid >= 16) {
            k1 = div(a_shared[tid | 32], - b_shared[47]);
            e = ((tid == 16) ?   b_shared[48] : ZERO(val_t)) + k1 * c_shared[47];
            d_shared[tid | 32] = d_shared[tid | 32]  + k1 * d_shared[47];
            if(tid == 16)  b_shared[48] = e;
        }
        __syncwarp();
        if(tid > 16) {
            k3 = div(e, - b_shared[48]);
            c_shared[tid | 32] = c_shared[tid | 32] + k3 * c_shared[48];
            d_shared[tid | 32] = d_shared[tid | 32] + k3 * d_shared[48];
        }
        c_shared[tid | 32] = div(c_shared[tid | 32], b_shared[tid | 32]);
        d_shared[tid | 32] = div(d_shared[tid | 32], b_shared[tid | 32]);
        c_mem[i + (tid | 32)] = c_shared[tid | 32];
        d_mem[i + (tid | 32)] = d_shared[tid | 32];
        __syncwarp();
        last_c = c_shared[63];
        last_d = d_shared[63];
    }
    val_t last_x = ZERO(val_t);
    for(int i = n_eqt-WARP_SIZE; i >= 0; i -= WARP_SIZE) {
        val_t c = - c_mem[i + tid], d = d_mem[i + tid];
        if(tid >= 16) {
            d_shared[tid] = d + last_x * c;
        }
        __syncwarp();
        if(tid < 16) {
            d_shared[tid] = d + d_shared[16] * c;
        }
        d_mem[i + tid] = d_shared[tid];
        __syncwarp();
        last_x = d_shared[0];
    }
}

template<typename val_t, int BLOCK_FUSION>
__global__ void WM5_reg_kernel(val_t *a_gbl, val_t *b_gbl, val_t *c_gbl, val_t *d_gbl, int n_eqt, int n_batch) {
    const unsigned int shuffle_mask = 0xffffffff;
    const int warp_size = warpSize;
    const int tid = threadIdx.x & 31;
    const int bid = (blockIdx.x * (BLOCK_FUSION * WARP_SIZE) + threadIdx.x) >> 5;
    if(bid >= n_batch) { return; }
    val_t *a_mem = a_gbl + bid*n_eqt;
    val_t *b_mem = b_gbl + bid*n_eqt;
    val_t *c_mem = c_gbl + bid*n_eqt;
    val_t *d_mem = d_gbl + bid*n_eqt;
    
    val_t last_c = ZERO(val_t), last_d = ZERO(val_t);
    for(int i = 0; i < n_eqt; i += 2 * WARP_SIZE) {
        val_t al, ah, bl, bh, cl, ch, dl, dh;
        auto a2 = load_2(a_mem + i + tid * 2);
        auto b2 = load_2(b_mem + i + tid * 2);
        auto c2 = load_2(c_mem + i + tid * 2);
        auto d2 = load_2(d_mem + i + tid * 2);
        al = a2.x; bl = b2.x; cl = c2.x; dl = d2.x;
        ah = a2.y; bh = b2.y; ch = c2.y; dh = d2.y;
        {
            int tidf, tidl, pred;
            val_t k1, k2, k3, e, all, ahf, bll, bhf, cll, chf, dll, dhf, at, bt, ct, dt;

            k1 = div(ah, - bl);
            bh = bh + k1 * cl;
            ah =      k1 * al;
            dh = dh + k1 * dl;
            k2 = div(cl, - bh);
            al = al + k2 * ah;
            cl =      k2 * ch;
            dl = dl + k2 * dh;
    
            pred = tid & 1;
            at = shfl_xor((pred == 0) ? ah : al, 1);
            bt = shfl_xor((pred == 0) ? bh : bl, 1);
            ct = shfl_xor((pred == 0) ? ch : cl, 1);
            dt = shfl_xor((pred == 0) ? dh : dl, 1);
            if(pred == 0) {
                ah = at, bh = bt, ch = ct, dh = dt;
            } else {
                al = at, bl = bt, cl = ct, dl = dt;
            }
    
            tidf = tid & (~1);
            tidl = tid | 1;
            all = shfl(al, tidl);
            bll = shfl(bl, tidl);
            cll = shfl(cl, tidl);
            dll = shfl(dl, tidl);
            k1 = div(ah, - bll);
            e = ((tid == tidf) ? bh : ZERO(val_t)) + k1 * cll;
            ah =      k1 * all;
            dh = dh + k1 * dll;
            bh = (tid == tidf) ? e : bh;
            ahf = shfl(ah, tidf);
            bhf = shfl(e, tidf);
            chf = shfl(ch, tidf);
            dhf = shfl(dh, tidf);
            k2 = div(cl, - bhf);
            k3 = maskz_div(tid == tidf, e, - bhf);
            al = al + k2 * ahf;
            cl =      k2 * chf;
            dl = dl + k2 * dhf;
            ah = ah + k3 * ahf;
            ch = ch + k3 * chf;
            dh = dh + k3 * dhf;
                
            pred = tid & 2;
            at = shfl_xor((pred == 0) ? ah : al, 2);
            bt = shfl_xor((pred == 0) ? bh : bl, 2);
            ct = shfl_xor((pred == 0) ? ch : cl, 2);
            dt = shfl_xor((pred == 0) ? dh : dl, 2);
            if(pred == 0) {
                ah = at, bh = bt, ch = ct, dh = dt;
            } else {
                al = at, bl = bt, cl = ct, dl = dt;
            }
            
            tidf = tid & (~3);
            tidl = tid | 3;
            all = shfl(al, tidl);
            bll = shfl(bl, tidl);
            cll = shfl(cl, tidl);
            dll = shfl(dl, tidl);
            k1 = div(ah, - bll);
            e = ((tid == tidf) ? bh : ZERO(val_t)) + k1 * cll;
            ah =      k1 * all;
            dh = dh + k1 * dll;
            bh = (tid == tidf) ? e : bh;
            ahf = shfl(ah, tidf);
            bhf = shfl(e, tidf);
            chf = shfl(ch, tidf);
            dhf = shfl(dh, tidf);
            k2 = div(cl, - bhf);
            k3 = maskz_div(tid == tidf, e, - bhf);
            al = al + k2 * ahf;
            cl =      k2 * chf;
            dl = dl + k2 * dhf;
            ah = ah + k3 * ahf;
            ch = ch + k3 * chf;
            dh = dh + k3 * dhf;
                
            pred = tid & 4;
            at = shfl_xor((pred == 0) ? ah : al, 4);
            bt = shfl_xor((pred == 0) ? bh : bl, 4);
            ct = shfl_xor((pred == 0) ? ch : cl, 4);
            dt = shfl_xor((pred == 0) ? dh : dl, 4);
            if(pred == 0) {
                ah = at, bh = bt, ch = ct, dh = dt;
            } else {
                al = at, bl = bt, cl = ct, dl = dt;
            }
            
            tidf = tid & (~7);
            tidl = tid | 7;
            all = shfl(al, tidl);
            bll = shfl(bl, tidl);
            cll = shfl(cl, tidl);
            dll = shfl(dl, tidl);
            k1 = div(ah, - bll);
            e = ((tid == tidf) ? bh : ZERO(val_t)) + k1 * cll;
            ah =      k1 * all;
            dh = dh + k1 * dll;
            bh = (tid == tidf) ? e : bh;
            ahf = shfl(ah, tidf);
            bhf = shfl(e, tidf);
            chf = shfl(ch, tidf);
            dhf = shfl(dh, tidf);
            k2 = div(cl, - bhf);
            k3 = maskz_div(tid == tidf, e, - bhf);
            al = al + k2 * ahf;
            cl =      k2 * chf;
            dl = dl + k2 * dhf;
            ah = ah + k3 * ahf;
            ch = ch + k3 * chf;
            dh = dh + k3 * dhf;
                
            pred = tid & 8;
            at = shfl_xor((pred == 0) ? ah : al, 8);
            bt = shfl_xor((pred == 0) ? bh : bl, 8);
            ct = shfl_xor((pred == 0) ? ch : cl, 8);
            dt = shfl_xor((pred == 0) ? dh : dl, 8);
            if(pred == 0) {
                ah = at, bh = bt, ch = ct, dh = dt;
            } else {
                al = at, bl = bt, cl = ct, dl = dt;
            }
            
            tidf = tid & (~15);
            tidl = tid | 15;
            all = shfl(al, tidl);
            bll = shfl(bl, tidl);
            cll = shfl(cl, tidl);
            dll = shfl(dl, tidl);
            k1 = div(ah, - bll);
            e = ((tid == tidf) ? bh : ZERO(val_t)) + k1 * cll;
            ah =      k1 * all;
            dh = dh + k1 * dll;
            bh = (tid == tidf) ? e : bh;
            ahf = shfl(ah, tidf);
            bhf = shfl(e, tidf);
            chf = shfl(ch, tidf);
            dhf = shfl(dh, tidf);
            k2 = div(cl, - bhf);
            k3 = maskz_div(tid == tidf, e, - bhf);
            al = al + k2 * ahf;
            cl =      k2 * chf;
            dl = dl + k2 * dhf;
            ah = ah + k3 * ahf;
            ch = ch + k3 * chf;
            dh = dh + k3 * dhf;
                
            pred = tid & 16;
            at = shfl_xor((pred == 0) ? ah : al, 16);
            bt = shfl_xor((pred == 0) ? bh : bl, 16);
            ct = shfl_xor((pred == 0) ? ch : cl, 16);
            dt = shfl_xor((pred == 0) ? dh : dl, 16);
            if(pred == 0) {
                ah = at, bh = bt, ch = ct, dh = dt;
            } else {
                al = at, bl = bt, cl = ct, dl = dt;
            }

            k1 = - al;
            e = ((tid == 0) ? bl : ZERO(val_t)) + k1 * last_c;
            dl = dl + k1 * last_d;
            bl = (tid == 0) ? e : bl;
            bhf = shfl(e, 0);
            chf = shfl(cl, 0);
            dhf = shfl(dl, 0);
            k3 = maskz_div(tid == 0, e, - bhf);
            cl = div(cl + k3 * chf, bl);
            dl = div(dl + k3 * dhf, bl);
            c_mem[tid + i] = cl;
            d_mem[tid + i] = dl;
            last_c = shfl(cl, 31);
            last_d = shfl(dl, 31);
    
            k1 = - ah;
            e = ((tid == 0) ? bh : ZERO(val_t)) + k1 * last_c;
            dh = dh + k1 * last_d;
            bh = (tid == 0) ? e : bh;
            bhf = shfl(e, 0);
            chf = shfl(ch, 0);
            dhf = shfl(dh, 0);
            k3 = maskz_div(tid == 0, e, - bhf);
            ch = div(ch + k3 * chf, bh);
            dh = div(dh + k3 * dhf, bh);
            c_mem[i + (tid | 32)] = ch;
            d_mem[i + (tid | 32)] = dh;
            last_c = shfl(ch, 31);
            last_d = shfl(dh, 31);
        }
    }
    val_t last_x = d_mem[n_eqt - WARP_SIZE];
    for(int i = n_eqt-2*WARP_SIZE; i >= 0; i -= WARP_SIZE) {
        val_t x = d_mem[i+tid] - last_x * c_mem[i+tid];
        d_mem[i+tid] = x;
        last_x = shfl(x, 0);
    }
}

template<typename val_t, int BLOCK_FUSION>
__global__ void WM4_reg_kernel(val_t *a_gbl, val_t *b_gbl, val_t *c_gbl, val_t *d_gbl, int n_eqt, int n_batch) {
    const int tid = threadIdx.x & 31;
    const int bid = (blockIdx.x * (BLOCK_FUSION * WARP_SIZE) + threadIdx.x) >> 5;
    const unsigned int shuffle_mask = 0xffffffff;
    const int warp_size = warpSize;
    if(bid >= n_batch) { return; }
    val_t *a_mem = a_gbl + bid*n_eqt;
    val_t *b_mem = b_gbl + bid*n_eqt;
    val_t *c_mem = c_gbl + bid*n_eqt;
    val_t *d_mem = d_gbl + bid*n_eqt;
    
    val_t last_c = ZERO(val_t), last_d = ZERO(val_t);
    for(int i = 0; i < n_eqt; i += 2 * WARP_SIZE) {
        val_t al, ah, bl, bh, cl, ch, dl, dh;
        auto a2 = load_2(a_mem + i + tid * 2);
        auto b2 = load_2(b_mem + i + tid * 2);
        auto c2 = load_2(c_mem + i + tid * 2);
        auto d2 = load_2(d_mem + i + tid * 2);
        al = a2.x; bl = b2.x; cl = c2.x; dl = d2.x;
        ah = a2.y; bh = b2.y; ch = c2.y; dh = d2.y;
        {
            int tidf, tidl, pred;
            val_t k1, k2, k3, e, all, ahf, bll, bhf, cll, chf, dll, dhf, at, bt, ct, dt;
            k1 = div(ah, - bl);
            bh = bh + k1 * cl;
            ah =    + k1 * al;
            dh = dh + k1 * dl;
            k2 = div(cl, - bh);
            al = al + k2 * ah;
            cl =    + k2 * ch;
            dl = dl + k2 * dh;
    
            pred = tid & 1;
            at = shfl_xor((pred == 0) ? ah : al, 1);
            bt = shfl_xor((pred == 0) ? bh : bl, 1);
            ct = shfl_xor((pred == 0) ? ch : cl, 1);
            dt = shfl_xor((pred == 0) ? dh : dl, 1);
            if(pred == 0) {
                ah = at, bh = bt, ch = ct, dh = dt;
            } else {
                al = at, bl = bt, cl = ct, dl = dt;
            }
            
            /* ROUND 2-4 */
            tidf = tid & (~1);
            tidl = tid | 1;
            all = shfl(al, tidl);
            bll = shfl(bl, tidl);
            cll = shfl(cl, tidl);
            dll = shfl(dl, tidl);
            k1 = div(ah, - bll);
            e = ((tid == tidf) ? bh : ZERO(val_t)) + k1 * cll;
            ah =      k1 * all;
            dh = dh + k1 * dll;
            bh = (tid == tidf) ? e : bh;
            ahf = shfl(ah, tidf);
            bhf = shfl(e, tidf);
            chf = shfl(ch, tidf);
            dhf = shfl(dh, tidf);
            k2 = div(cl, - bhf);
            k3 = maskz_div(tid == tidf, e, - bhf);
            al = al + k2 * ahf;
            cl =      k2 * chf;
            dl = dl + k2 * dhf;
            ah = ah + k3 * ahf;
            ch = ch + k3 * chf;
            dh = dh + k3 * dhf;
                
            pred = tid & 2;
            at = shfl_xor((pred == 0) ? ah : al, 2);
            bt = shfl_xor((pred == 0) ? bh : bl, 2);
            ct = shfl_xor((pred == 0) ? ch : cl, 2);
            dt = shfl_xor((pred == 0) ? dh : dl, 2);
            if(pred == 0) {
                ah = at, bh = bt, ch = ct, dh = dt;
            } else {
                al = at, bl = bt, cl = ct, dl = dt;
            }
            
            tidf = tid & (~3);
            tidl = tid | 3;
            all = shfl(al, tidl);
            bll = shfl(bl, tidl);
            cll = shfl(cl, tidl);
            dll = shfl(dl, tidl);
            k1 = div(ah, - bll);
            e = ((tid == tidf) ? bh : ZERO(val_t)) + k1 * cll;
            ah =      k1 * all;
            dh = dh + k1 * dll;
            bh = (tid == tidf) ? e : bh;
            ahf = shfl(ah, tidf);
            bhf = shfl(e, tidf);
            chf = shfl(ch, tidf);
            dhf = shfl(dh, tidf);
            k2 = div(cl, - bhf);
            k3 = maskz_div(tid == tidf, e, - bhf);
            al = al + k2 * ahf;
            cl =      k2 * chf;
            dl = dl + k2 * dhf;
            ah = ah + k3 * ahf;
            ch = ch + k3 * chf;
            dh = dh + k3 * dhf;
                
            pred = tid & 4;
            at = shfl_xor((pred == 0) ? ah : al, 4);
            bt = shfl_xor((pred == 0) ? bh : bl, 4);
            ct = shfl_xor((pred == 0) ? ch : cl, 4);
            dt = shfl_xor((pred == 0) ? dh : dl, 4);
            if(pred == 0) {
                ah = at, bh = bt, ch = ct, dh = dt;
            } else {
                al = at, bl = bt, cl = ct, dl = dt;
            }
            
            tidf = tid & (~7);
            tidl = tid | 7;
            all = shfl(al, tidl);
            bll = shfl(bl, tidl);
            cll = shfl(cl, tidl);
            dll = shfl(dl, tidl);
            k1 = div(ah, - bll);
            e = ((tid == tidf) ? bh : ZERO(val_t)) + k1 * cll;
            ah =      k1 * all;
            dh = dh + k1 * dll;
            bh = (tid == tidf) ? e : bh;
            ahf = shfl(ah, tidf);
            bhf = shfl(e, tidf);
            chf = shfl(ch, tidf);
            dhf = shfl(dh, tidf);
            k2 = div(cl, - bhf);
            k3 = maskz_div(tid == tidf, e, - bhf);
            al = al + k2 * ahf;
            cl =      k2 * chf;
            dl = dl + k2 * dhf;
            ah = ah + k3 * ahf;
            ch = ch + k3 * chf;
            dh = dh + k3 * dhf;
                
            pred = tid & 8;
            at = shfl_xor((pred == 0) ? ah : al, 8);
            bt = shfl_xor((pred == 0) ? bh : bl, 8);
            ct = shfl_xor((pred == 0) ? ch : cl, 8);
            dt = shfl_xor((pred == 0) ? dh : dl, 8);
            if(pred == 0) {
                ah = at, bh = bt, ch = ct, dh = dt;
            } else {
                al = at, bl = bt, cl = ct, dl = dt;
            }
            
            /* ROUND 5,DO NOTHING */
            pred = tid & 16;
            at = shfl_xor((pred == 0) ? ah : al, 16);
            bt = shfl_xor((pred == 0) ? bh : bl, 16);
            ct = shfl_xor((pred == 0) ? ch : cl, 16);
            dt = shfl_xor((pred == 0) ? dh : dl, 16);
            if(pred == 0) {
                ah = at, bh = bt, ch = ct, dh = dt;
            } else {
                al = at, bl = bt, cl = ct, dl = dt;
            }
  
            {
                val_t btl, ctl, dtl, bth, cth, dth, bf, cf, df, e;
                k1 = - al;
                e = ((tid == 0) ? bl : ZERO(val_t)) + k1 * last_c;
                dtl = dl + k1 * last_d;
                btl = (tid == 0) ? e : bl;
                bf = shfl(e, 0);
                cf = shfl(cl, 0);
                df = shfl(dtl, 0);
                k3 = maskz_div(tid == 0, e, - bf);
                ctl = cl  + k3 * cf;
                dtl = dtl + k3 * df;
                last_c = shfl(ctl, 15);
                last_d = shfl(dtl, 15);
        
                k1 = div(al, - shfl(btl, 15));
                e = ((tid == 16) ? bl : ZERO(val_t)) + k1 * last_c;
                dth = dl + k1 * last_d;
                bth = (tid == 16) ? e : bl;
                bf = shfl(e, 16);
                cf = shfl(cl, 16);
                df = shfl(dth, 16);
                k3 = maskz_div(tid == 16, e, - bf);
                cth = cl  + k3 * cf;
                dth = dth + k3 * df;
                bl = (tid < 16) ? btl : bth;
                cl = div((tid < 16) ? ctl : cth, bl);
                dl = div((tid < 16) ? dtl : dth, bl);
                c_mem[i + tid] = cl;
                d_mem[i + tid] = dl;
                last_c = shfl(cl, 31);
                last_d = shfl(dl, 31);
        
                k1 = - ah;
                e = ((tid == 0) ? bh : ZERO(val_t)) + k1 * last_c;
                dtl = dh + k1 * last_d;
                btl = (tid == 0) ? e : bh;
                bf = shfl(e, 0);
                cf = shfl(ch, 0);
                df = shfl(dtl, 0);
                k3 = maskz_div(tid == 0, e, - bf);
                ctl = ch  + k3 * cf;
                dtl = dtl + k3 * df;
                last_c = shfl(ctl, 15);
                last_d = shfl(dtl, 15);
        
                k1 = div(ah, - shfl(btl, 15));
                e = ((tid == 16) ? bh : ZERO(val_t)) + k1 * last_c;
                dth = dh + k1 * last_d;
                bth = (tid == 16) ? e : bh;
                bf = shfl(e, 16);
                cf = shfl(ch, 16);
                df = shfl(dth, 16);
                k3 = maskz_div(tid == 16, e, - bf);
                cth = ch  + k3 * cf;
                dth = dth + k3 * df;
                bh = (tid < 16) ? btl : bth;
                ch = div((tid < 16) ? ctl : cth, bh);
                dh = div((tid < 16) ? dtl : dth, bh);
                c_mem[i + tid + 32] = ch;
                d_mem[i + tid + 32] = dh;
                last_c = shfl(ch, 31);
                last_d = shfl(dh, 31);
            }
        }
    }
    val_t last_x = ZERO(val_t);
    for(int i = n_eqt-WARP_SIZE; i >= 0; i -= WARP_SIZE) {
        val_t xl, xh, c = - c_mem[i + tid], d = d_mem[i + tid];
        xh = d + last_x * c;
        xl = d + shfl(xh, 16) * c;
        d_mem[i + tid] = (tid < 16) ? xl : xh;
        last_x = shfl(xl, 0);
    }
}

template<typename val_t, int BLOCK_FUSION>
__global__ void PCR5_reg_kernel(val_t *a_gbl, val_t *b_gbl, val_t *c_gbl, val_t *d_gbl, int n_eqt, int n_batch) {
    const int tid = threadIdx.x & 31;
    const int bid = (blockIdx.x * (BLOCK_FUSION * WARP_SIZE) + threadIdx.x) >> 5;
    const unsigned int shuffle_mask = 0xffffffff;
    const int warp_size = warpSize;
    if(bid >= n_batch) { return; }
    
    val_t aa0, aa1, aa2;
    val_t bb0, bb1, bb2;
    val_t cc0, cc1, cc2;
    val_t dd0, dd1, dd2;
    val_t *a_mem = a_gbl + bid*n_eqt;
    val_t *b_mem = b_gbl + bid*n_eqt;
    val_t *c_mem = c_gbl + bid*n_eqt;
    val_t *d_mem = d_gbl + bid*n_eqt;
    aa0 = ZERO(val_t); aa1 = ZERO(val_t); aa2 = ZERO(val_t);
    bb0 = ONE(val_t);  bb1 = ONE(val_t);  bb2 = ONE(val_t);
    cc0 = ZERO(val_t); cc1 = ZERO(val_t); cc2 = ZERO(val_t);
    dd0 = ZERO(val_t); dd1 = ZERO(val_t); dd2 = ZERO(val_t);
  
     val_t ccc = ZERO(val_t), ddd = ZERO(val_t);

    for(int i = 0; i < n_eqt+WARP_SIZE; i += WARP_SIZE) {
        if(i < n_eqt) {
            aa2 = a_mem[tid+i];
            bb2 = b_mem[tid+i];
            cc2 = c_mem[tid+i];
            dd2 = d_mem[tid+i];
        } else {
            aa2 = ZERO(val_t);
            bb2 = ONE(val_t);
            cc2 = ZERO(val_t);
            dd2 = ZERO(val_t);
        }
        val_t k1, k2, a, b, c, d, ta, tb, tc, td;
        {
            k1 = div(concatenate(aa2, aa1, tid, 30), - concatenate(bb2, bb1, tid, 29));
            k2 = div(concatenate(cc2, cc1, tid, 30), - concatenate(bb2, bb1, tid, 31));
            a =                                  k1*concatenate(aa2, aa1, tid, 29);
            b = concatenate(bb2, bb1, tid, 30) + k1*concatenate(cc2, cc1, tid, 29) + k2*concatenate(aa2, aa1, tid, 31);
            c =                                                                      k2*concatenate(cc2, cc1, tid, 31);
            d = concatenate(dd2, dd1, tid, 30) + k1*concatenate(dd2, dd1, tid, 29) + k2*concatenate(dd2, dd1, tid, 31);
            ta = shfl(a, tid+4);
            tb = shfl(b, tid+4);
            tc = shfl(c, tid+4);
            td = shfl(d, tid+4);
            aa1 = ((tid<28)?aa1:ta); aa2 = ((tid<28)?ta:aa2);
            bb1 = ((tid<28)?bb1:tb); bb2 = ((tid<28)?tb:bb2);
            cc1 = ((tid<28)?cc1:tc); cc2 = ((tid<28)?tc:cc2);
            dd1 = ((tid<28)?dd1:td); dd2 = ((tid<28)?td:dd2);
            ta = a;
            tb = b;
            tc = c;
            td = d;

            k1 = div(concatenate(aa2, aa1, tid, 26), - concatenate(bb2, bb1, tid, 24));
            k2 = div(concatenate(cc2, cc1, tid, 26), - tb);
            a =                                 k1*concatenate(aa2, aa1, tid, 24);
            b = concatenate(bb2,bb1, tid, 26) + k1*concatenate(cc2, cc1, tid, 24) + k2*ta;
            c =                                                                     k2*tc;
            d = concatenate(dd2,dd1, tid, 26) + k1*concatenate(dd2, dd1, tid, 24) + k2*td;
            ta = shfl(a, tid+8);
            tb = shfl(b, tid+8);
            tc = shfl(c, tid+8);
            td = shfl(d, tid+8);
            aa1 = ((tid<24)?aa1:ta); aa2 = ((tid<24)?ta:aa2);
            bb1 = ((tid<24)?bb1:tb); bb2 = ((tid<24)?tb:bb2);
            cc1 = ((tid<24)?cc1:tc); cc2 = ((tid<24)?tc:cc2);
            dd1 = ((tid<24)?dd1:td); dd2 = ((tid<24)?td:dd2);
            ta = a;
            tb = b;
            tc = c;
            td = d;

            k1 = div(concatenate(aa2, aa1, tid, 20), - concatenate(bb2, bb1, tid, 16));
            k2 = div(concatenate(cc2, cc1, tid, 20), - tb);
            a =                                  k1*concatenate(aa2, aa1, tid, 16);
            b = concatenate(bb2, bb1, tid, 20) + k1*concatenate(cc2, cc1, tid, 16) + k2*ta;
            c =                                                                      k2*tc;
            d = concatenate(dd2, dd1, tid, 20) + k1*concatenate(dd2, dd1, tid, 16) + k2*td;
            ta = shfl(a, tid+16);
            tb = shfl(b, tid+16);
            tc = shfl(c, tid+16);
            td = shfl(d, tid+16);
            aa1 = ((tid<16)?aa1:ta); aa2 = ((tid<16)?ta:aa2);
            bb1 = ((tid<16)?bb1:tb); bb2 = ((tid<16)?tb:bb2);
            cc1 = ((tid<16)?cc1:tc); cc2 = ((tid<16)?tc:cc2);
            dd1 = ((tid<16)?dd1:td); dd2 = ((tid<16)?td:dd2);
            ta = a;
            tb = b;
            tc = c;
            td = d;

            k1 = div(concatenate(aa2, aa1, tid, 8), - bb1);
            k2 = div(concatenate(cc2, cc1, tid, 8), - tb);
            a =                                 k1*aa1;
            b = concatenate(bb2, bb1, tid, 8) + k1*cc1 + k2*ta;
            c =                                          k2*tc;
            d = concatenate(dd2, dd1, tid, 8) + k1*dd1 + k2*td;
            aa1 = a;
            bb1 = b;
            cc1 = c;
            dd1 = d;
        }
        if(i != 0) {
            k1 = div(concatenate(aa1, aa0, tid, 16), - bb0);
            k2 = div(concatenate(cc1, cc0, tid, 16), - bb1);
            a =                                  k1*aa0;
            b = concatenate(bb1, bb0, tid, 16) + k1*cc0 + k2*aa1;
            c =                                           k2*cc1;
            d = concatenate(dd1, dd0, tid, 16) + k1*dd0 + k2*dd1;

            val_t an = - a;
            val_t kk = (b+ccc*an);
            ccc = div(c,        kk);
            ddd = div(d+ddd*an, kk);
            c_mem[tid+i-WARP_SIZE] = ccc;
            d_mem[tid+i-WARP_SIZE] = ddd;
        }
        aa0 = aa1; aa1 = aa2;
        bb0 = bb1; bb1 = bb2;
        cc0 = cc1; cc1 = cc2;
        dd0 = dd1; dd1 = dd2;
    }
    val_t r = d_mem[tid + n_eqt - WARP_SIZE];
    for(int i = n_eqt-2*WARP_SIZE; i >= 0; i -= WARP_SIZE) {
        val_t d = d_mem[tid+i];
        val_t c = c_mem[tid+i];
        r = d - c*r;
        d_mem[tid+i] = r; 
    }
}

template<typename val_t, int BLOCK_FUSION>
__global__ void PCR4_reg_kernel(val_t *a_gbl, val_t *b_gbl, val_t *c_gbl, val_t *d_gbl, int n_eqt, int n_batch) {
    const int tid = threadIdx.x & 31;
    const int bid = (blockIdx.x * (BLOCK_FUSION * WARP_SIZE) + threadIdx.x) >> 5;
    const unsigned int shuffle_mask = 0xffffffff;
    const int warp_size = warpSize;
    if(bid >= n_batch) { return; }
    val_t aa0, aa1;
    val_t bb0, bb1;
    val_t cc0, cc1;
    val_t dd0, dd1;
    val_t *a_mem = a_gbl + bid*n_eqt;
    val_t *b_mem = b_gbl + bid*n_eqt;
    val_t *c_mem = c_gbl + bid*n_eqt;
    val_t *d_mem = d_gbl + bid*n_eqt;
    aa0 = ZERO(val_t); aa1 = ZERO(val_t);
    bb0 = ONE(val_t);  bb1 = ONE(val_t);
    cc0 = ZERO(val_t); cc1 = ZERO(val_t);
    dd0 = ZERO(val_t); dd1 = ZERO(val_t);
  
    val_t ccc = ZERO(val_t), ddd = ZERO(val_t);
    for(int i = 0; i < n_eqt+WARP_SIZE; i += WARP_SIZE)
    {
        if(i == 0) {
            aa1 = ((tid < 16)?ZERO(val_t):a_mem[tid+i-16]);
            bb1 = ((tid < 16)?ONE(val_t):b_mem[tid+i-16]);
            cc1 = ((tid < 16)?ZERO(val_t):c_mem[tid+i-16]);
            dd1 = ((tid < 16)?ZERO(val_t):d_mem[tid+i-16]);
        } else if(i < n_eqt) {
            aa1 = a_mem[tid+i-16];
            bb1 = b_mem[tid+i-16];
            cc1 = c_mem[tid+i-16];
            dd1 = d_mem[tid+i-16];
        } else {
            aa1 = ((tid < 16)?a_mem[tid+i-16]:ZERO(val_t));
            bb1 = ((tid < 16)?b_mem[tid+i-16]:ONE(val_t));
            cc1 = ((tid < 16)?c_mem[tid+i-16]:ZERO(val_t));
            dd1 = ((tid < 16)?d_mem[tid+i-16]:ZERO(val_t));
        }
        val_t k1, k2, a, b, c, d, ta, tb, tc, td;
        {
            k1 = div(concatenate(aa1, aa0, tid, 30), - concatenate(bb1, bb0, tid, 29));
            k2 = div(concatenate(cc1, cc0, tid, 30), - concatenate(bb1, bb0, tid, 31));
            a =                                  k1*concatenate(aa1, aa0, tid, 29);
            b = concatenate(bb1, bb0, tid, 30) + k1*concatenate(cc1, cc0, tid, 29) + k2*concatenate(aa1, aa0, tid, 31);
            c =                                                                      k2*concatenate(cc1, cc0, tid, 31);
            d = concatenate(dd1, dd0, tid, 30) + k1*concatenate(dd1, dd0, tid, 29) + k2*concatenate(dd1, dd0, tid, 31);
            ta = shfl(a, tid+4);
            tb = shfl(b, tid+4);
            tc = shfl(c, tid+4);
            td = shfl(d, tid+4);
            aa0 = ((tid<28)?aa0:ta); aa1 = ((tid<28)?ta:aa1);
            bb0 = ((tid<28)?bb0:tb); bb1 = ((tid<28)?tb:bb1);
            cc0 = ((tid<28)?cc0:tc); cc1 = ((tid<28)?tc:cc1);
            dd0 = ((tid<28)?dd0:td); dd1 = ((tid<28)?td:dd1);
            ta = a;
            tb = b;
            tc = c;
            td = d;

            k1 = div(concatenate(aa1, aa0, tid, 26), - concatenate(bb1, bb0, tid, 24));
            k2 = div(concatenate(cc1, cc0, tid, 26), - tb);
            a =                                  k1*concatenate(aa1, aa0, tid, 24);
            b = concatenate(bb1, bb0, tid, 26) + k1*concatenate(cc1, cc0, tid, 24) + k2*ta;
            c =                                                                      k2*tc;
            d = concatenate(dd1, dd0, tid, 26) + k1*concatenate(dd1, dd0, tid, 24) + k2*td;
            ta = shfl(a, tid+8);
            tb = shfl(b, tid+8);
            tc = shfl(c, tid+8);
            td = shfl(d, tid+8);
            aa0 = ((tid<24)?aa0:ta); aa1 = ((tid<24)?ta:aa1);
            bb0 = ((tid<24)?bb0:tb); bb1 = ((tid<24)?tb:bb1);
            cc0 = ((tid<24)?cc0:tc); cc1 = ((tid<24)?tc:cc1);
            dd0 = ((tid<24)?dd0:td); dd1 = ((tid<24)?td:dd1);
            ta = a;
            tb = b;
            tc = c;
            td = d;

            k1 = div(concatenate(aa1, aa0, tid, 20), - concatenate(bb1, bb0, tid, 16));
            k2 = div(concatenate(cc1, cc0, tid, 20), - tb);
            a =                                  k1*concatenate(aa1, aa0, tid, 16);
            b = concatenate(bb1, bb0, tid, 20) + k1*concatenate(cc1, cc0, tid, 16) + k2*ta;
            c =                                                                      k2*tc;
            d = concatenate(dd1, dd0, tid, 20) + k1*concatenate(dd1, dd0, tid, 16) + k2*td;
            ta = shfl(a, tid+16);
            tb = shfl(b, tid+16);
            tc = shfl(c, tid+16);
            td = shfl(d, tid+16);
            aa0 = ((tid<16)?aa0:ta); aa1 = ((tid<16)?ta:aa1);
            bb0 = ((tid<16)?bb0:tb); bb1 = ((tid<16)?tb:bb1);
            cc0 = ((tid<16)?cc0:tc); cc1 = ((tid<16)?tc:cc1);
            dd0 = ((tid<16)?dd0:td); dd1 = ((tid<16)?td:dd1);
            ta = a;
            tb = b;
            tc = c;
            td = d;
        }
        if(i != 0) {
            k1 = div(concatenate(aa1, aa0, tid, 8), - bb0);
            k2 = div(concatenate(cc1, cc0, tid, 8), - tb);
            a =                                 k1*aa0;
            b = concatenate(bb1, bb0, tid, 8) + k1*cc0 + k2*ta;
            c =                                          k2*tc;
            d = concatenate(dd1, dd0, tid, 8) + k1*dd0 + k2*td;
            
            val_t kk, cct, ddt, an = -a;
            kk = (b+ccc*an);
            ccc = div(c,       kk);
            ddd = div(d+ddd*an, kk);
            cct = shfl(ccc, tid+16);
            ddt = shfl(ddd, tid+16);
            kk = (b+cct*an);
            cct = div(c,       kk);
            ddt = div(d+ddt*an, kk);
            ccc = ((tid<16)?ccc:cct);
            ddd = ((tid<16)?ddd:ddt);
            c_mem[tid+i-WARP_SIZE] = ccc;
            d_mem[tid+i-WARP_SIZE] = ddd;
            ccc = shfl(ccc, tid+16);
            ddd = shfl(ddd, tid+16);
        }
        aa0 = aa1;
        bb0 = bb1;
        cc0 = cc1;
        dd0 = dd1;
    }
    val_t r = ZERO(val_t);
    for(int i = n_eqt-WARP_SIZE; i >= 0; i -= WARP_SIZE) {
        val_t d = d_mem[tid+i];
        val_t c = - c_mem[tid+i];
        r  = d + c*r;
        val_t rt = shfl(r, tid+16);
        rt = d + c*rt;
        r = ((tid<16)?rt:r);
        d_mem[tid+i] = r;
        r = shfl(r, tid+16);
    }
}

template<typename val_t, int BLOCK_FUSION>
__global__ void PCR5_shmem_kernel(val_t *a_gbl, val_t *b_gbl, val_t *c_gbl, val_t *d_gbl, int n_eqt, int n_batch) {
    const int tid = threadIdx.x & 31;
    const int bid = (blockIdx.x * (BLOCK_FUSION * WARP_SIZE) + threadIdx.x) >> 5;
    if(bid >= n_batch) { return; }
    
    __shared__ val_t aa_shared[BLOCK_FUSION][96];
    __shared__ val_t bb_shared[BLOCK_FUSION][96];
    __shared__ val_t cc_shared[BLOCK_FUSION][96];
    __shared__ val_t dd_shared[BLOCK_FUSION][96];
    val_t *a_shared = aa_shared[threadIdx.x >> 5];
    val_t *b_shared = bb_shared[threadIdx.x >> 5];
    val_t *c_shared = cc_shared[threadIdx.x >> 5];
    val_t *d_shared = dd_shared[threadIdx.x >> 5];
    val_t *a_mem = a_gbl + bid*n_eqt;
    val_t *b_mem = b_gbl + bid*n_eqt;
    val_t *c_mem = c_gbl + bid*n_eqt;
    val_t *d_mem = d_gbl + bid*n_eqt;
    a_shared[tid] = ZERO(val_t); a_shared[tid+32] = ZERO(val_t);
    b_shared[tid] = ONE(val_t); b_shared[tid+32] = ONE(val_t);
    c_shared[tid] = ZERO(val_t); c_shared[tid+32] = ZERO(val_t);
    d_shared[tid] = ZERO(val_t); d_shared[tid+32] = ZERO(val_t);
    __syncwarp();

    val_t cc0 = ZERO(val_t), dd0 = ZERO(val_t);
    for(int i = 0; i < n_eqt+WARP_SIZE; i += WARP_SIZE) {
        if(i < n_eqt) {
            a_shared[tid+64] = a_mem[tid+i];
            b_shared[tid+64] = b_mem[tid+i];
            c_shared[tid+64] = c_mem[tid+i];
            d_shared[tid+64] = d_mem[tid+i];
        } else {
            a_shared[tid+64] = ZERO(val_t);
            b_shared[tid+64] = ONE(val_t);
            c_shared[tid+64] = ZERO(val_t);
            d_shared[tid+64] = ZERO(val_t);
        }
        __syncwarp();
        {
            val_t k1, k2, a, b, c, d;
            k1 = div(a_shared[tid+62], - b_shared[tid+61]);
            k2 = div(c_shared[tid+62], - b_shared[tid+63]);
            a =                    k1*a_shared[tid+61];;
            b = b_shared[tid+62] + k1*c_shared[tid+61] + k2*a_shared[tid+63];
            c =                                          k2*c_shared[tid+63];
            d = d_shared[tid+62] + k1*d_shared[tid+61] + k2*d_shared[tid+63];
            __syncwarp();
            a_shared[tid+60] = a;
            b_shared[tid+60] = b;
            c_shared[tid+60] = c;
            d_shared[tid+60] = d;
            __syncwarp();

            k1 = div(a_shared[tid+58], - b_shared[tid+56]);
            k2 = div(c_shared[tid+58], - b_shared[tid+60]);
            a =                    k1*a_shared[tid+56];
            b = b_shared[tid+58] + k1*c_shared[tid+56] + k2*a_shared[tid+60];
            c =                                          k2*c_shared[tid+60];
            d = d_shared[tid+58] + k1*d_shared[tid+56] + k2*d_shared[tid+60];
            __syncwarp();
            a_shared[tid+56] = a;
            b_shared[tid+56] = b;
            c_shared[tid+56] = c;
            d_shared[tid+56] = d;
            __syncwarp();

            k1 = div(a_shared[tid+52], - b_shared[tid+48]);
            k2 = div(c_shared[tid+52], - b_shared[tid+56]);
            a =                    k1*a_shared[tid+48];;
            b = b_shared[tid+52] + k1*c_shared[tid+48] + k2*a_shared[tid+56];
            c =                                          k2*c_shared[tid+56];
            d = d_shared[tid+52] + k1*d_shared[tid+48] + k2*d_shared[tid+56];
            __syncwarp();
            a_shared[tid+48] = a;
            b_shared[tid+48] = b;
            c_shared[tid+48] = c;
            d_shared[tid+48] = d;
            __syncwarp();

            k1 = div(a_shared[tid+40], - b_shared[tid+32]);
            k2 = div(c_shared[tid+40], - b_shared[tid+48]);
            a =                    k1*a_shared[tid+32];;
            b = b_shared[tid+40] + k1*c_shared[tid+32] + k2*a_shared[tid+48];
            c =                                          k2*c_shared[tid+48];
            d = d_shared[tid+40] + k1*d_shared[tid+32] + k2*d_shared[tid+48];
            __syncwarp();
            a_shared[tid+32] = a;
            b_shared[tid+32] = b;
            c_shared[tid+32] = c;
            d_shared[tid+32] = d;
            __syncwarp();
            if(i >= WARP_SIZE)
            {
                k1 = div(a_shared[tid+16], - b_shared[tid+0 ]);
                k2 = div(c_shared[tid+16], - b_shared[tid+32]);
                a =                    k1*a_shared[tid+0 ];
                b = b_shared[tid+16] + k1*c_shared[tid+0 ] + k2*a_shared[tid+32];
                c =                                          k2*c_shared[tid+32];
                d = d_shared[tid+16] + k1*d_shared[tid+0 ] + k2*d_shared[tid+32];
                __syncwarp();

                val_t an = -a;
                val_t kk = (b+cc0*an);
                cc0 = div(c,       kk);
                dd0 = div(d+dd0*an, kk);
                c_mem[tid+i-WARP_SIZE] = cc0;
                d_mem[tid+i-WARP_SIZE] = dd0;
            }
        }
        a_shared[tid] = a_shared[tid+32]; a_shared[tid+32] = a_shared[tid+64];
        b_shared[tid] = b_shared[tid+32]; b_shared[tid+32] = b_shared[tid+64];
        c_shared[tid] = c_shared[tid+32]; c_shared[tid+32] = c_shared[tid+64];
        d_shared[tid] = d_shared[tid+32]; d_shared[tid+32] = d_shared[tid+64];
        __syncwarp();
    }
    val_t r = d_mem[n_eqt-WARP_SIZE+tid];
    for(int i = n_eqt-2*WARP_SIZE; i >= 0; i -= WARP_SIZE) {
        val_t d = d_mem[tid+i];
        val_t c = c_mem[tid+i];
        r = d - c*r;
        d_mem[tid+i] = r; 
    }
}

template<typename val_t, int BLOCK_FUSION>
__global__ void PCR4_shmem_kernel(val_t *a_gbl, val_t *b_gbl, val_t *c_gbl, val_t *d_gbl, int n_eqt, int n_batch) {
    const int tid = threadIdx.x & 31;
    const int bid = (blockIdx.x * (BLOCK_FUSION * WARP_SIZE) + threadIdx.x) >> 5;
    if(bid >= n_batch) { return; }
    __shared__ val_t aa_shared[BLOCK_FUSION][64];
    __shared__ val_t bb_shared[BLOCK_FUSION][64];
    __shared__ val_t cc_shared[BLOCK_FUSION][64];
    __shared__ val_t dd_shared[BLOCK_FUSION][64];
    val_t *a_shared = aa_shared[threadIdx.x >> 5];
    val_t *b_shared = bb_shared[threadIdx.x >> 5];
    val_t *c_shared = cc_shared[threadIdx.x >> 5];
    val_t *d_shared = dd_shared[threadIdx.x >> 5];
    val_t *a_mem = a_gbl + bid*n_eqt;
    val_t *b_mem = b_gbl + bid*n_eqt;
    val_t *c_mem = c_gbl + bid*n_eqt;
    val_t *d_mem = d_gbl + bid*n_eqt;
    a_shared[tid] = ZERO(val_t);
    b_shared[tid] = ONE(val_t);
    c_shared[tid] = ZERO(val_t);
    d_shared[tid] = ZERO(val_t);
    __syncwarp();
    
    val_t cc0 = ZERO(val_t), dd0 = ZERO(val_t);

    for(int i = 0; i < n_eqt+WARP_SIZE; i += WARP_SIZE)
    {
        if(i == 0) {
            a_shared[tid+32] = ((tid < 16)?ZERO(val_t):a_mem[tid+i-16]);
            b_shared[tid+32] = ((tid < 16)?ONE(val_t):b_mem[tid+i-16]);
            c_shared[tid+32] = ((tid < 16)?ZERO(val_t):c_mem[tid+i-16]);
            d_shared[tid+32] = ((tid < 16)?ZERO(val_t):d_mem[tid+i-16]);
        } else if(i < n_eqt) {
            a_shared[tid+32] = a_mem[tid+i-16];
            b_shared[tid+32] = b_mem[tid+i-16];
            c_shared[tid+32] = c_mem[tid+i-16];
            d_shared[tid+32] = d_mem[tid+i-16];
        } else {
            a_shared[tid+32] = ((tid < 16)?a_mem[tid+i-16]:ZERO(val_t));
            b_shared[tid+32] = ((tid < 16)?b_mem[tid+i-16]:ONE(val_t));
            c_shared[tid+32] = ((tid < 16)?c_mem[tid+i-16]:ZERO(val_t));
            d_shared[tid+32] = ((tid < 16)?d_mem[tid+i-16]:ZERO(val_t));
        }
        __syncwarp();

        {
            val_t k1, k2, a, b, c, d;
            k1 = div(a_shared[tid+30], - b_shared[tid+29]);
            k2 = div(c_shared[tid+30], - b_shared[tid+31]);
            a =                    k1*a_shared[tid+29];
            b = b_shared[tid+30] + k1*c_shared[tid+29] + k2*a_shared[tid+31];
            c =                                          k2*c_shared[tid+31];
            d = d_shared[tid+30] + k1*d_shared[tid+29] + k2*d_shared[tid+31];
            __syncwarp();
            a_shared[tid+28] = a;
            b_shared[tid+28] = b;
            c_shared[tid+28] = c;
            d_shared[tid+28] = d;
            __syncwarp();

            k1 = div(a_shared[tid+26], - b_shared[tid+24]);
            k2 = div(c_shared[tid+26], - b_shared[tid+28]);
            a =                    k1*a_shared[tid+24];
            b = b_shared[tid+26] + k1*c_shared[tid+24] + k2*a_shared[tid+28];
            c =                                          k2*c_shared[tid+28];
            d = d_shared[tid+26] + k1*d_shared[tid+24] + k2*d_shared[tid+28];
            __syncwarp();
            a_shared[tid+24] = a;
            b_shared[tid+24] = b;
            c_shared[tid+24] = c;
            d_shared[tid+24] = d;
            __syncwarp();

            k1 = div(a_shared[tid+20], - b_shared[tid+16]);
            k2 = div(c_shared[tid+20], - b_shared[tid+24]);
            a =                    k1*a_shared[tid+16];
            b = b_shared[tid+20] + k1*c_shared[tid+16] + k2*a_shared[tid+24];
            c =                                          k2*c_shared[tid+24];
            d = d_shared[tid+20] + k1*d_shared[tid+16] + k2*d_shared[tid+24];
            __syncwarp();
            a_shared[tid+16] = a;
            b_shared[tid+16] = b;
            c_shared[tid+16] = c;
            d_shared[tid+16] = d;
            __syncwarp();

            if(i != 0) {
                k1 = div(a_shared[tid+8], - b_shared[tid+0]);
                k2 = div(c_shared[tid+8], - b_shared[tid+16]);
                a =                   k1*a_shared[tid+0];;
                b = b_shared[tid+8] + k1*c_shared[tid+0] + k2*a_shared[tid+16];
                c =                                        k2*c_shared[tid+16];
                d = d_shared[tid+8] + k1*d_shared[tid+0] + k2*d_shared[tid+16];
                __syncwarp();
                val_t an = -a;
                if(tid < 16) {
                    val_t kk = (b+cc0*an);
                    cc0 = div(c, kk);
                    dd0 = div(d+dd0*an, kk);
                    c_shared[tid] = cc0;
                    d_shared[tid] = dd0;
                }
                __syncwarp();
                if(tid >= 16) {
                    val_t kk = (b+c_shared[tid-16]*an);
                    cc0 = div(c, kk);
                    dd0 = div(d+d_shared[tid-16]*an, kk);
                    c_shared[tid - 16] = cc0;
                    d_shared[tid - 16] = dd0;
                }
                __syncwarp();
                c_mem[tid+i-WARP_SIZE] = cc0;
                d_mem[tid+i-WARP_SIZE] = dd0;
                if(tid < 16) {
                    cc0 = c_shared[tid];
                    dd0 = d_shared[tid];
                }
                __syncwarp();
            }
        }
        a_shared[tid] = a_shared[tid+32];
        b_shared[tid] = b_shared[tid+32];
        c_shared[tid] = c_shared[tid+32];
        d_shared[tid] = d_shared[tid+32];
        __syncwarp();
    }
    val_t x = ZERO(val_t);
    for(int i = n_eqt-WARP_SIZE; i >= 0; i -= WARP_SIZE)
    {
        val_t d = d_mem[tid+i];
        val_t c = - c_mem[tid+i];
        
        if(tid >= 16) {
            x = d + c * x;
            d_shared[tid - 16] = x;
        }
        __syncwarp();
        if(tid < 16) {
            x = d + c * d_shared[tid];
            d_shared[tid] = x; 
        }
        __syncwarp();
        d_mem[tid+i] = x;
        if(tid >= 16) {
            x = d_shared[tid - 16];
        }
    }
}

inline int div_ceil(int a, int b) {
    return (a + 1) / b - 1;
}

#define F_WM5_shmem_float   2
#define F_WM4_shmem_float   2
#define F_PCR5_shmem_float  2
#define F_PCR4_shmem_float  2
#define F_WM5_reg_float     2
#define F_WM4_reg_float     2
#define F_PCR5_reg_float    2
#define F_PCR4_reg_float    2
#define F_WM5_shmem_double  2
#define F_WM4_shmem_double  2
#define F_PCR5_shmem_double 2
#define F_PCR4_shmem_double 2
#define F_WM5_reg_double    2
#define F_WM4_reg_double    2
#define F_PCR5_reg_double   2
#define F_PCR4_reg_double   2

#define KERNEL(NAME, TYPE) NAME##_kernel<TYPE, F_##NAME##_##TYPE>
#define CALL(NAME, TYPE, a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch) KERNEL(NAME, TYPE)<<<div_ceil(n_batch, F_##NAME##_##TYPE), F_##NAME##_##TYPE * WARP_SIZE>>>(a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch)
#define EXPLICIT_LAUNCH_BOUNDS(NAME, TYPE, NUM) \
template __global__ void __launch_bounds__(F_##NAME##_##TYPE * WARP_SIZE, NUM) \
KERNEL(NAME, TYPE)(TYPE *a_gbl, TYPE *b_gbl, TYPE *c_gbl, TYPE *d_gbl, int n_eqt, int n_batch);

EXPLICIT_LAUNCH_BOUNDS(PCR5_reg, double, 8)
EXPLICIT_LAUNCH_BOUNDS(PCR4_reg, double, 8)
EXPLICIT_LAUNCH_BOUNDS(PCR5_reg, float, 8)
EXPLICIT_LAUNCH_BOUNDS(PCR4_reg, float, 8)
EXPLICIT_LAUNCH_BOUNDS(WM5_reg, double, 8)
EXPLICIT_LAUNCH_BOUNDS(WM4_reg, double, 8)
EXPLICIT_LAUNCH_BOUNDS(WM5_reg, float, 8)
EXPLICIT_LAUNCH_BOUNDS(WM4_reg, float, 8)

void WM5_shmem_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch) {
    CALL(WM5_shmem, float, a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
    cudaDeviceSynchronize();
}
void WM4_shmem_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch) {
    CALL(WM4_shmem, float, a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
    cudaDeviceSynchronize();
}
void PCR5_shmem_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch) {
    CALL(PCR5_shmem, float, a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
    cudaDeviceSynchronize();
}
void PCR4_shmem_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch) {
    CALL(PCR4_shmem, float, a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
    cudaDeviceSynchronize();
}

void WM5_reg_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch) {
    CALL(WM5_reg, float, a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
    cudaDeviceSynchronize();
}
void WM4_reg_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch) {
    CALL(WM4_reg, float, a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
    cudaDeviceSynchronize();
}

void PCR5_reg_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch) {
    CALL(PCR5_reg, float, a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
    cudaDeviceSynchronize();
}
void PCR4_reg_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch) {
    CALL(PCR4_reg, float, a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
    cudaDeviceSynchronize();
}

void WM5_shmem_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch) {
    CALL(WM5_shmem, double, a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
    cudaDeviceSynchronize();
}
void WM4_shmem_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch) {
    CALL(WM4_shmem, double, a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
    cudaDeviceSynchronize();
}
void PCR5_shmem_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch) {
    CALL(PCR5_shmem, double, a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
    cudaDeviceSynchronize();
}
void PCR4_shmem_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch) {
    CALL(PCR4_shmem, double, a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
    cudaDeviceSynchronize();
}

void WM5_reg_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch) {
    CALL(WM5_reg, double, a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
    cudaDeviceSynchronize();
}
void WM4_reg_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch) {
    CALL(WM4_reg, double, a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
    cudaDeviceSynchronize();
}

void PCR5_reg_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch) {
    CALL(PCR5_reg, double, a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
    cudaDeviceSynchronize();
}
void PCR4_reg_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch) {
    CALL(PCR4_reg, double, a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
    cudaDeviceSynchronize();
}

class TridiagonalConfig {
public:
    TridiagonalConfig() {
        cudaFuncSetSharedMemConfig(KERNEL(WM5_shmem,  float),   cudaSharedMemBankSizeFourByte);
        cudaFuncSetSharedMemConfig(KERNEL(WM4_shmem,  float),   cudaSharedMemBankSizeFourByte);
        cudaFuncSetSharedMemConfig(KERNEL(PCR5_shmem, float),   cudaSharedMemBankSizeFourByte);
        cudaFuncSetSharedMemConfig(KERNEL(PCR4_shmem, float),   cudaSharedMemBankSizeFourByte);
        cudaFuncSetSharedMemConfig(KERNEL(WM5_shmem,  double),  cudaSharedMemBankSizeEightByte);
        cudaFuncSetSharedMemConfig(KERNEL(WM4_shmem,  double),  cudaSharedMemBankSizeEightByte);
        cudaFuncSetSharedMemConfig(KERNEL(PCR5_shmem, double),  cudaSharedMemBankSizeEightByte);
        cudaFuncSetSharedMemConfig(KERNEL(PCR4_shmem, double),  cudaSharedMemBankSizeEightByte);
        cudaFuncSetCacheConfig(KERNEL(WM5_shmem,  float),   cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(KERNEL(WM4_shmem,  float),   cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(KERNEL(PCR5_shmem, float),   cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(KERNEL(PCR4_shmem, float),   cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(KERNEL(WM5_shmem,  double),  cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(KERNEL(WM4_shmem,  double),  cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(KERNEL(PCR5_shmem, double),  cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(KERNEL(PCR4_shmem, double),  cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(KERNEL(WM5_reg,  float),   cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(KERNEL(WM4_reg,  float),   cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(KERNEL(PCR5_reg, float),   cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(KERNEL(PCR4_reg, float),   cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(KERNEL(WM5_reg,  double),  cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(KERNEL(WM4_reg,  double),  cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(KERNEL(PCR5_reg, double),  cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(KERNEL(PCR4_reg, double),  cudaFuncCachePreferL1);
    }
};

static const TridiagonalConfig config;
