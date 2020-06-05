#include "tridiagonal.h"

template<typename val_t>
__global__ void Thomas_kernel(val_t *a_gbl, val_t *b_gbl, val_t *c_gbl, val_t *d_gbl, int n_eqt, int n_batch) {
    int task_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(task_id < n_batch){
        int i;
        val_t *a = a_gbl + task_id*n_eqt;
        val_t *b = b_gbl + task_id*n_eqt;
        val_t *c = c_gbl + task_id*n_eqt;
        val_t *r = d_gbl + task_id*n_eqt;
        val_t cc = c[0]/b[0];
        val_t rr = r[0]/b[0];
        c[0] = cc;
        r[0] = rr;

        for(i = 1; i < n_eqt; i ++) {
            val_t k = (b[i] - cc*a[i]);
            cc = c[i] / k;
            rr = (r[i] - rr*a[i]) / k;
            c[i] = cc;
            r[i] = rr;
        }
        for(i = n_eqt-2; i >= 0; i --) {
            rr = r[i] - c[i] * rr;
            r[i] = rr;
        }
    }
}

template<typename val_t>
__global__ void CR_kernel(val_t *a_gbl, val_t *b_gbl, val_t *c_gbl, val_t *d_gbl, val_t *buffer, int n_eqt, int n_batch) {
    const int tid = threadIdx.x;
    const int tnum = blockDim.x;
    const int bid = blockIdx.x;
    
    val_t *a = a_gbl + bid*n_eqt;
    val_t *b = b_gbl + bid*n_eqt;
    val_t *c = c_gbl + bid*n_eqt;
    val_t *d = d_gbl + bid*n_eqt;
    
    val_t *a1 = a;
    val_t *b1 = b;
    val_t *c1 = c;
    val_t *d1 = d;

    val_t *a2 = buffer + 0*n_eqt*n_batch + bid*n_eqt;
    val_t *b2 = buffer + 1*n_eqt*n_batch + bid*n_eqt;
    val_t *c2 = buffer + 2*n_eqt*n_batch + bid*n_eqt;
    val_t *d2 = buffer + 3*n_eqt*n_batch + bid*n_eqt;
    
    __shared__ int length_record[32];
    int iter = 0;
    if(tid==0) length_record[0] = n_eqt;
    __syncthreads();
    //CR Forward
    while(length_record[iter] >= 2*tnum) {
        int n = length_record[iter]/2;
        if(length_record[iter]%2)
        {
            int i = tid;
            for(i = tid; i < n; i += tnum) {
                val_t k1 = a1[2*i+1]/b1[2*i-0];
                val_t k2 = c1[2*i+1]/b1[2*i+2];
                a2[i] = -a1[2*i-0] * k1;
                b2[i] = b1[2*i+1] - c1[2*i-0]*k1 - a1[2*i+2]*k2;
                c2[i] = -c1[2*i+2] * k2;
                d2[i] = d1[2*i+1] - d1[2*i-0]*k1 - d1[2*i+2]*k2;
            }
        } else {
            int i = tid;
            {
                val_t k1 = a1[2*i]/((tid==0)?1:b1[2*i-1]);
                val_t k2 = c1[2*i]/b1[2*i+1];
                a2[i] = -((tid==0)?0:a1[2*i-1]) * k1;
                b2[i] = b1[2*i] - ((tid==0)?0:c1[2*i-1])*k1 - a1[2*i+1]*k2;
                c2[i] = -c1[2*i+1] * k2;
                d2[i] = d1[2*i] - ((tid==0)?0:d1[2*i-1])*k1 - d1[2*i+1]*k2;
            }
            for(i = tid+tnum; i < n; i += tnum) {
                val_t k1 = a1[2*i]/b1[2*i-1];
                val_t k2 = c1[2*i]/b1[2*i+1];
                a2[i] = -a1[2*i-1] * k1;
                b2[i] = b1[2*i] - c1[2*i-1]*k1 - a1[2*i+1]*k2;
                c2[i] = -c1[2*i+1] * k2;
                d2[i] = d1[2*i] - d1[2*i-1]*k1 - d1[2*i+1]*k2;
            }
        }
        a1 = a2;
        b1 = b2;
        c1 = c2;
        d1 = d2;
        a2 = a2 + n;
        b2 = b2 + n;
        c2 = c2 + n;
        d2 = d2 + n;
        ++iter;
        if(tid == 0)length_record[iter] = n;
        __syncthreads();
    }
    //Thomas
    if(tid == 0) {
        int n = length_record[iter];
        c1[0] = c1[0]/b1[0];
        d1[0] = d1[0]/b1[0];
        int i;
        for(i = 1; i < n; i ++) {
            val_t k = b1[i]-c1[i-1]*a1[i];
            c1[i] = c1[i] / k;
            d1[i] = (d1[i]-d1[i-1]*a1[i]) / k;
        }
        for(i = n-2; i >= 0; i --) {
            d1[i] = d1[i] - c1[i]*d1[i+1];
        }
    }
    __syncthreads();

    iter --;
    while(iter >= 0) {
        d2 = d1;
        if(iter > 0) {
            a1 = a1 - length_record[iter];
            b1 = b1 - length_record[iter];
            c1 = c1 - length_record[iter];
            d1 = d1 - length_record[iter];
        } else {
            a1 = a;
            b1 = b;
            c1 = c;
            d1 = d;
        }
        int n = length_record[iter]/2;
        if(length_record[iter]%2) {
            int ii = tid;
            {
                int i = n - 1 - ii;
                d1[2*i+1] = d2[i];
                d1[2*i+2] = (d1[2*i+2] - a1[2*i+2]*d2[i] - c1[2*i+2]*((tid==0)?0:d2[i+1]))/b1[2*i+2];
            }
            for(ii = tid+tnum; ii < n; ii += tnum) {
                int i = n - 1 - ii;
                d1[2*i+1] = d2[i];
                d1[2*i+2] = (d1[2*i+2] - a1[2*i+2]*d2[i] - c1[2*i+2]*d2[i+1])/b1[2*i+2];
            }
            if(tid == 0)
                d1[0] = (d1[0] - c1[0]*d2[0])/b1[0];
        } else {
            int ii = tid;
            {
                int i = n - 1 - ii;
                d1[2*i] = d2[i];
                d1[2*i+1] = (d1[2*i+1] - a1[2*i+1]*d2[i] - c1[2*i+1]*((tid==0)?0:d2[i+1]))/b1[2*i+1];
            }
            for(ii = tid+tnum; ii < n; ii += tnum) {
                int i = n - 1 - ii;
                d1[2*i] = d2[i];
                d1[2*i+1] = (d1[2*i+1] - a1[2*i+1]*d2[i] - c1[2*i+1]*d2[i+1])/b1[2*i+1];
            }
        }
        iter --;
        __syncthreads();
    }
}

#define THOMAS_THREAD_NUM 64
#define CR_THREAD_NUM 64

static void *buffer;

void Thomas_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch) {
    int block_num = (n_batch-1) / THOMAS_THREAD_NUM + 1;
    Thomas_kernel<float><<<block_num, THOMAS_THREAD_NUM>>>(a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
    cudaDeviceSynchronize();
}

void CR_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch) {
    CR_kernel<float><<<n_batch, CR_THREAD_NUM>>>(a_gbl, b_gbl, c_gbl, d_gbl, static_cast<float*>(buffer), n_eqt, n_batch);
    cudaDeviceSynchronize();
}

void Thomas_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch) {
    int block_num = (n_batch-1) / THOMAS_THREAD_NUM + 1;
    Thomas_kernel<double><<<block_num, THOMAS_THREAD_NUM>>>(a_gbl, b_gbl, c_gbl, d_gbl, n_eqt, n_batch);
    cudaDeviceSynchronize();
}

void CR_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch) {
    CR_kernel<double><<<n_batch, CR_THREAD_NUM>>>(a_gbl, b_gbl, c_gbl, d_gbl, static_cast<double*>(buffer), n_eqt, n_batch);
    cudaDeviceSynchronize();
}

int CR_init_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch) {
    return cudaMalloc(&buffer, 4 * n_eqt * n_batch * sizeof(float));
}

int CR_init_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch) {
    return cudaMalloc(&buffer, 4 * n_eqt * n_batch * sizeof(double));
}

void CR_final() {
    cudaFree(buffer);
}
