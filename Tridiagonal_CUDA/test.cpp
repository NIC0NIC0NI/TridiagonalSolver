#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <mkl.h>
#include "tridiagonal.h"
#include "helpers.h"
#include "utils.h"

#define RANDSEED 13541532
#define MKL_RAND MKL_rand_uniform

using namespace std;
void print_mkl_version();
void print_help(const char *argv0);
void print_algorithms();

template<typename val_t>
void tridiag_gen(val_t *a, val_t *b, val_t *c, val_t *d, val_t *x, int n) {
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, RANDSEED);
    MKL_RAND(stream, n - 1, a + 1, -1.0, 1.0);
    MKL_RAND(stream, n - 1, c,     -1.0, 1.0);
    MKL_RAND(stream, n,     x,     -1.0, 1.0);
    MKL_RAND(stream, n,     b,      0.0, 1.0);
    vslDeleteStream(&stream);
    a[0] = 0.0;
    c[n-1] = 0.0;
    b[0] += max(fabs(c[0]), fabs(a[1]));
    d[0] = b[0] * x[0] + c[0] * x[1];
    for(int i = 1; i < n - 1; ++i) {
        b[i] += max(fabs(a[i]) + fabs(c[i]), fabs(a[i+1]) + fabs(c[i-1]));
        d[i] = a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1];
    }
    b[n-1] += max(fabs(a[n-1]), fabs(c[n-2]));
    d[n-1] = a[n-1] * x[n-2] + b[n-1] * x[n-1];
}


template<typename val_t>
void rcondest(val_t *a, val_t *b, val_t *c, val_t *d, int *ipiv, int n, val_t *rcond) {
    val_t norm1, norminf;
    norm1 = fabs(a[1]) + fabs(b[0]);
    norminf = fabs(b[0]) + fabs(c[0]);
    for(int i = 1; i < n - 1; ++i) {
        norm1 = max(norm1, fabs(a[i+1]) + fabs(b[i]) + fabs(c[i-1]));
        norminf = max(norminf, fabs(a[i]) + fabs(b[i]) + fabs(c[i]));
    }
    norm1 = max(norm1, fabs(b[n-1]) + fabs(c[n-2]));
    norminf = max(norminf, fabs(a[n-1]) + fabs(b[n-1]));
    MKL_rcondest(a+1, b, c, d, ipiv, n, norm1, norminf, rcond);
}

template<typename T>
inline int error(int code, const T & msg) {
    cerr << msg << endl;
    return code;
}

template<typename val_t>
int test(int algo, int size, int batch, int rep, const char *out) {
    if(algo >= arr_len(selector<val_t>::solvers)) {
        print_algorithms();
        return 1;
    } else {
        const solver_t<val_t> *solver = &selector<val_t>::solvers[algo];
        const char *name = solver->name;
        val_t *original = (val_t*)malloc(size*sizeof(val_t)*5);
        val_t *arr      = (val_t*)cuda_malloc(size*sizeof(val_t)*4*batch);
        val_t *a_original = original + 0*size;
        val_t *b_original = original + 1*size;
        val_t *c_original = original + 2*size;
        val_t *d_original = original + 3*size;
        val_t *x_original = original + 4*size;
        val_t *a_batch = arr + 0*size*batch;
        val_t *b_batch = arr + 1*size*batch;
        val_t *c_batch = arr + 2*size*batch;
        val_t *d_batch = arr + 3*size*batch;
        val_t stats[5], cond1, condinf, err_norm1, err_norm2, err_norminf;
        double begin, t_total = 0.0, t, throughput;

        if(original == NULL)  return error(1, "Not enough memory");
        if(arr == NULL)       return error(1, "Not enough GPU memory");
        if(solver->init != nullptr) {
            int ret = solver->init(a_batch, b_batch, c_batch, d_batch, size, batch);
            if(ret != 0)  return error(1, "Initialization failed");
        }

        cout << name << endl;

        tridiag_gen<val_t>(a_original, b_original, c_original, d_original, x_original, size);

        for(int i = 0; i < rep; ++i) {
            init_batch(a_original, b_original, c_original, d_original, \
                            a_batch, b_batch, c_batch, d_batch, size, batch, solver->layout);
            begin = dsecnd();
            solver->routine(a_batch, b_batch, c_batch, d_batch, size, batch);
            t_total += dsecnd() - begin;
        }
        
        int err_maxid = check(d_batch, x_original, a_batch, b_batch, reinterpret_cast<int*>(c_batch), size, stats, solver->layout);
        rcondest(a_original, b_original, c_original, d_original, \
                reinterpret_cast<int*>(x_original), size, stats + 3);

        free(original); cuda_free(arr); 
        if(solver->final != nullptr) { solver->final(); }

        err_norm1 = stats[0];
        err_norm2 = stats[1];
        err_norminf = stats[2];
        cond1 = 1.0 / stats[3];
        condinf = 1.0 / stats[4];
        t = t_total / rep;
        throughput = 1e-9 * size * batch * rep / t_total;
        cout << scientific << "  1-norm condition number: " << cond1 << ", inf-norm condition number: " << condinf << endl << "Cheking error:" << endl \
            << "Average: " << err_norm1 << endl \
            << "RMSE:    " << err_norm2 << endl \
            << "Max:     " << err_norminf << "  (max position: " << err_maxid << ")" << endl \
            << fixed << "Size:" << setw(7) << size << ",   Time: " << setprecision(3) << (t * 1e3) << " ms,   " \
            << "Throughput: " << setprecision(6) << throughput << " N/ns" << endl << endl;
        if(out != NULL) {
            ofstream(out, ios_base::app) << setw(6) << size << ',' << setw(8) << batch << ',' \
                << setw(25) << name << ',' \
                << setw(15) << setprecision(6) << t << ',' \
                << setw(15) << setprecision(6) << throughput << ',' << scientific \
                << setw(15) << setprecision(6) << cond1 << ',' \
                << setw(15) << setprecision(6) << condinf << ',' \
                << setw(15) << setprecision(6) << err_norm1 << ',' \
                << setw(15) << setprecision(6) << err_norm2 << ',' \
                << setw(15) << setprecision(6) << err_norminf << ',' \
                << setw(6) << err_maxid << endl;
        }
        return 0;
    }
}

typedef int (*tester_t)(int, int, int, int, const char*);
static const tester_t tester[] = { test<float>, test<double> };

int main(int argc, char **argv) {
    ios_base::sync_with_stdio(false);
    if(argc < 6) { print_help(argv[0]); print_algorithms(); return 1; }
    else {
        int prec = atoi(argv[1]);
        int algo = atoi(argv[2]);
        int size = atoi(argv[3]);
        int batch = atoi(argv[4]);
        int rep = atoi(argv[5]);
        if(prec < 0 || prec > 1 || algo < 0 || size <= 0 || batch <= 0 || rep <= 0) {
            print_help(argv[0]); print_algorithms(); return 1;
        }
        return tester[prec](algo, size, batch, rep, (argc > 6) ? argv[6] : NULL);
    }
}

void print_algorithms() {
    int n_algo = arr_len(selector<float>::solvers);
    cout << "single-precision algorithms:" << endl;
    for(int i = 0; i < n_algo; ++i) {
        cout << setw(3) << i << ": " << selector<float>::solvers[i].name << endl;
    }
    cout << "double-precision algorithms:" << endl;
    n_algo = arr_len(selector<double>::solvers);
    for(int i = 0; i < n_algo; ++i) {
        cout << setw(3) << i << ": " << selector<double>::solvers[i].name << endl;
    }
}

void print_help(const char *argv0) {
    cout << "USAGE: " << argv0 << " <precision> <algorithm> <size> <batch> <repetitions> [out file]" << endl;
}

