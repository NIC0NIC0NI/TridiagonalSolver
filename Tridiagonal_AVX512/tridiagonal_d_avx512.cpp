#include "tridiagonal.h"
#include "threading.h"
#include "vec.h"

#define VECLEN    8

#define MASK0145  0x33
#define MASK2367  0xCC //_mm512_knot(MASK0145)
#define MASK0123  0x0F
#define MASK4567  0xF0 //_mm512_knot(MASK0123)
#define MASK0246  0x55
#define MASK1357  0xaa //_mm512_knot(MASK0246)
#define MASKN0246 0xaa //_mm512_knot(MASK0246)
#define MASK04    0x11
#define MASKN04   0xee //_mm512_knot(MASK04)
#define MASK0     0x01
#define MASKN0    0xfe //_mm512_knot(MASK0)
#define MASK4     0x10
#define MASK123   0x0e
#define MASK567   0xe0

#define F2I(x)   _mm512_castpd_si512(x)
#define I2F(x)   _mm512_castsi512_pd(x)

/* x[offset], ..., x[7], y[0], ..., y[offset-1] */
#define CONCATENATE(x, y, offset) I2F(_mm512_alignr_epi64(F2I(y), F2I(x), offset))
#define CYCLICSHIFT(x, offset)    I2F(_mm512_alignr_epi64(F2I(x), F2I(x), offset))

#define BROADCASTD0246(x)  _mm512_movedup_pd(x)            /* _mm512_permute_pd(ah, 0x00) */
#define BROADCASTD1357(x)  _mm512_permute_pd(x, 0xff)
#define BROADCASTD04(x)    _mm512_permutex_pd(x, 0x00)
#define BROADCASTD37(x)    _mm512_permutex_pd(x, 0xff)
#define BROADCASTD0(x)     _mm512_broadcastsd_pd(_mm512_castpd512_pd128(x))
#define BROADCASTD3(x)     _mm512_permutexvar_pd(idx3bcast, x)
#define BROADCASTD4(x)     _mm512_permutexvar_pd(idx4bcast, x)
#define BROADCASTD7(x)     _mm512_permutexvar_pd(idx7bcast, x)

/* READ to ROUND1 */
/* x[0],x[2],x[4],x[6],y[0],y[2],y[4],y[6] */
#define PACKD0246(x, y)     _mm512_permutex2var_pd(x, idxl, y)
#define PACKD1357(x, y)     _mm512_permutex2var_pd(x, idxh, y)

/* ROUND1 to ROUND2 */
/* x[0],y[0],x[2],y[2],x[4],y[4],x[6],y[6] */
#ifdef USE_SHUFFLE
#define SHUFFLED0246(x, y)  _mm512_shuffle_pd(x, y, 0x00)
#define SHUFFLED1357(x, y)  _mm512_shuffle_pd(x, y, 0xff)
#else
/* valignq is faster than vunpcklpd, vunpckhpd, vshufpd and vshuff64x2, but uses an additional mask */
#define SHUFFLED0246(x, y)  _mm512_mask_movedup_pd(x, MASK1357, y)
#define SHUFFLED1357(x, y)  _mm512_mask_permute_pd(y, MASK0246, x, 0xff)
#endif

/* ROUND2 to ROUND3 */
/* x[0],x[1],y[0],y[1],x[4],x[5],y[4],y[5] */
#define SHUFFLE2D0145(x, y) _mm512_mask_permutex_pd(x, MASK2367, y, 0x44)
#define SHUFFLE2D2367(x, y) _mm512_mask_permutex_pd(y, MASK0145, x, 0xee)

/* ROUND3 to pGE */
/* x[0],x[1],x[2],x[3],y[0],y[1],y[2],y[3] */
#ifdef USE_SHUFFLE
#define PACKD0123(x, y)    _mm512_shuffle_f64x2(x, y, 0x44)
#define PACKD4567(x, y)    _mm512_shuffle_f64x2(x, y, 0xee)
#else
#define PACKD0123(x, y)     I2F(_mm512_mask_alignr_epi64(F2I(x), MASK4567, F2I(y), F2I(y), 4))
#define PACKD4567(x, y)     I2F(_mm512_mask_alignr_epi64(F2I(y), MASK0123, F2I(x), F2I(x), 4))
#endif

/* ROUND2 to pGE */
/* x[0],x[1],y[0],y[1],x[2],x[3],y[2],y[3] */
#define SHUFFLE2D0123(x, y) _mm512_permutex2var_pd(x, idx8l, y)
#define SHUFFLE2D4567(x, y) _mm512_permutex2var_pd(x, idx8h, y)

void WM3_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, double *buff, int n_eqt, int n_batch) {
    PARALLEL_FOR_BEGIN(j, 0, n_batch)
        const __m512i idx7bcast = _mm512_set1_epi64(7);
        const __m512i idxl = _mm512_set_epi64(14,12,10,8,6,4,2,0);
        const __m512i idxh = _mm512_set_epi64(15,13,11,9,7,5,3,1);
        double *a_mem = a_gbl + j * n_eqt;
        double *b_mem = b_gbl + j * n_eqt;
        double *c_mem = c_gbl + j * n_eqt;
        double *d_mem = d_gbl + j * n_eqt;
        __m512d last_c = _mm512_setzero_pd(), last_d = _mm512_setzero_pd();
        for (int i = 0; i < n_eqt; i += VECLEN * 2) {
            /* l: low, h: high, ll: low last, hf: high first */
            __m512d al, ah, al_old, all, ahf;
            __m512d bl, bh, bl_old, bll, bhf;
            __m512d cl, ch, cl_old, cll, chf;
            __m512d dl, dh, dl_old, dll, dhf;
            __m512d k1, k2, k3, e;

            al = _mm512_load_pd(a_mem + i);
            bl = _mm512_load_pd(b_mem + i);
            cl = _mm512_load_pd(c_mem + i);
            dl = _mm512_load_pd(d_mem + i);
            ah = _mm512_load_pd(a_mem + i + VECLEN);
            bh = _mm512_load_pd(b_mem + i + VECLEN);
            ch = _mm512_load_pd(c_mem + i + VECLEN);
            dh = _mm512_load_pd(d_mem + i + VECLEN);

                
            al_old = al; bl_old = bl; cl_old = cl; dl_old = dl;
            al = PACKD0246(al, ah);
            bl = PACKD0246(bl, bh);
            cl = PACKD0246(cl, ch);
            dl = PACKD0246(dl, dh);
            ah = PACKD1357(al_old, ah);
            bh = PACKD1357(bl_old, bh);
            ch = PACKD1357(cl_old, ch);
            dh = PACKD1357(dl_old, dh);

            /* ROUND 1 */
            k1 = div(ah, bl);
            bh = bh - k1 * cl;
            ah = nmul(k1,  al);
            dh = dh - k1 * dl;
            k2 = div(cl, bh);
            al = al - k2 * ah;
            cl = nmul(k2,  ch);
            dl = dl - k2 * dh;
            
            al_old = al; bl_old = bl; cl_old = cl; dl_old = dl;
            al = SHUFFLED0246(al, ah);
            bl = SHUFFLED0246(bl, bh);
            cl = SHUFFLED0246(cl, ch);
            dl = SHUFFLED0246(dl, dh);
            ah = SHUFFLED1357(al_old, ah);
            bh = SHUFFLED1357(bl_old, bh);
            ch = SHUFFLED1357(cl_old, ch);
            dh = SHUFFLED1357(dl_old, dh);
            
            /* ROUND 2 */
            all = BROADCASTD1357(al);
            bll = BROADCASTD1357(bl);
            cll = BROADCASTD1357(cl);
            dll = BROADCASTD1357(dl);
            k1  = div(ah, bll);
            e   = _mm512_maskz_mov_pd(MASK0246, bh);
            e   = e  - k1 * cll;
            ah  = nmul(k1,  all);
            dh  = dh - k1 * dll;
            bh  = _mm512_mask_mov_pd(bh, MASK0246, e);
            ahf = BROADCASTD0246(ah);
            bhf = BROADCASTD0246(e);
            chf = BROADCASTD0246(ch);
            dhf = BROADCASTD0246(dh);
            k2  = div(cl, bhf);
            k3  = maskz_div(MASKN0246, e, bhf);
            al  = al - k2 * ahf;
            cl  = nmul(k2,  chf);
            dl  = dl - k2 * dhf;
            ah  = ah - k3 * ahf;
            ch  = ch - k3 * chf;
            dh  = dh - k3 * dhf;

            al_old = al; bl_old = bl; cl_old = cl; dl_old = dl;
            al = SHUFFLE2D0145(al, ah);
            bl = SHUFFLE2D0145(bl, bh);
            cl = SHUFFLE2D0145(cl, ch);
            dl = SHUFFLE2D0145(dl, dh);
            ah = SHUFFLE2D2367(al_old, ah);
            bh = SHUFFLE2D2367(bl_old, bh);
            ch = SHUFFLE2D2367(cl_old, ch);
            dh = SHUFFLE2D2367(dl_old, dh);

            /* ROUND 3 */
            all = BROADCASTD37(al);
            bll = BROADCASTD37(bl);
            cll = BROADCASTD37(cl);
            dll = BROADCASTD37(dl);
            k1  = div(ah, bll);
            e   = _mm512_maskz_mov_pd(MASK04, bh);
            e   = e  - k1 * cll;
            ah  = nmul(k1,  all);
            dh  = dh - k1 * dll;
            bh  = _mm512_mask_mov_pd(bh, MASK04, e);
            ahf = BROADCASTD04(ah);
            bhf = BROADCASTD04(e);
            chf = BROADCASTD04(ch);
            dhf = BROADCASTD04(dh);
            k2  = div(cl, bhf);
            k3  = maskz_div(MASKN04, e, bhf);
            al  = al - k2 * ahf;
            cl  = nmul(k2,  chf);
            dl  = dl - k2 * dhf;
            ah  = ah - k3 * ahf;
            ch  = ch - k3 * chf;
            dh  = dh - k3 * dhf;

            al_old = al; bl_old = bl; cl_old = cl; dl_old = dl;
            al = PACKD0123(al, ah);
            bl = PACKD0123(bl, bh);
            cl = PACKD0123(cl, ch);
            dl = PACKD0123(dl, dh);
            ah = PACKD4567(al_old, ah);
            bh = PACKD4567(bl_old, bh);
            ch = PACKD4567(cl_old, ch);
            dh = PACKD4567(dl_old, dh);

            /* parallel Gaussian Elimination */
            e   = _mm512_maskz_mov_pd(MASK0, bl);
            e   = e  - al * last_c;
            dl  = dl - al * last_d;
            bl  = _mm512_mask_mov_pd(bl, MASK0, e);
            bhf = BROADCASTD0(e);
            chf = BROADCASTD0(cl);
            dhf = BROADCASTD0(dl);
            k3  = maskz_div(MASKN0, e, bhf);
            cl  = div(cl - k3 * chf, bl);
            dl  = div(dl - k3 * dhf, bl);
            _mm512_store_pd(c_mem + i, cl);
            _mm512_store_pd(d_mem + i, dl);
            last_c = BROADCASTD7(cl);
            last_d = BROADCASTD7(dl);

            e   = _mm512_maskz_mov_pd(MASK0, bh);
            e   = e  - ah * last_c;
            dh  = dh - ah * last_d;
            bh  = _mm512_mask_mov_pd(bh, MASK0, e);
            bhf = BROADCASTD0(e);
            chf = BROADCASTD0(ch);
            dhf = BROADCASTD0(dh);
            k3  = maskz_div(MASKN0, e, bhf);
            ch  = div(ch - k3 * chf, bh);
            dh  = div(dh - k3 * dhf, bh);
            _mm512_store_pd(c_mem + i + 8, ch);
            _mm512_store_pd(d_mem + i + 8, dh);
            last_c = BROADCASTD7(ch);
            last_d = BROADCASTD7(dh);
        }
        __m512d last_x = BROADCASTD0(_mm512_load_pd(d_mem + n_eqt - VECLEN));
        for(int i = n_eqt - VECLEN * 2; i >= 0; i -= VECLEN) {
            __m512d c, d, x;
            c = _mm512_load_pd(c_mem + i);
            d = _mm512_load_pd(d_mem + i);
            x = d - c * last_x;
            _mm512_store_pd(d_mem + i, x);
            last_x = BROADCASTD0(x);
        }
    PARALLEL_FOR_END
}

void WM2_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, double *buff, int n_eqt, int n_batch) {
    PARALLEL_FOR_BEGIN(j, 0, n_batch)
        const __m512i idx3bcast = _mm512_set1_epi64(3);
        const __m512i idx4bcast = _mm512_set1_epi64(4);
        const __m512i idx7bcast = _mm512_set1_epi64(7);
        const __m512i idxl = _mm512_set_epi64(14,12,10,8,6,4,2,0);
        const __m512i idxh = _mm512_set_epi64(15,13,11,9,7,5,3,1);
        const __m512i idx8l = _mm512_set_epi64(11,10,3,2,9,8,1,0);
        const __m512i idx8h = _mm512_set_epi64(15,14,7,6,13,12,5,4);
        double *a_mem = a_gbl + j * n_eqt;
        double *b_mem = b_gbl + j * n_eqt;
        double *c_mem = c_gbl + j * n_eqt;
        double *d_mem = d_gbl + j * n_eqt;
        __m512d last_c = _mm512_setzero_pd(), last_d = _mm512_setzero_pd();
        for (int i = 0; i < n_eqt; i += 16) {
            /* l: low, h: high, ll: low last, hf: high first */
            __m512d al, ah, al_old, all, ahf;
            __m512d bl, bh, bl_old, bll, bhf;
            __m512d cl, ch, cl_old, cll, chf;
            __m512d dl, dh, dl_old, dll, dhf;
            __m512d k1, k2, k3, e;

            al = _mm512_load_pd(a_mem + i);
            bl = _mm512_load_pd(b_mem + i);
            cl = _mm512_load_pd(c_mem + i);
            dl = _mm512_load_pd(d_mem + i);
            ah = _mm512_load_pd(a_mem + i + VECLEN);
            bh = _mm512_load_pd(b_mem + i + VECLEN);
            ch = _mm512_load_pd(c_mem + i + VECLEN);
            dh = _mm512_load_pd(d_mem + i + VECLEN);
                
            al_old = al; bl_old = bl; cl_old = cl; dl_old = dl;
            al = PACKD0246(al, ah);
            bl = PACKD0246(bl, bh);
            cl = PACKD0246(cl, ch);
            dl = PACKD0246(dl, dh);
            ah = PACKD1357(al_old, ah);
            bh = PACKD1357(bl_old, bh);
            ch = PACKD1357(cl_old, ch);
            dh = PACKD1357(dl_old, dh);
            
            /* ROUND 1 */
            k1 = div(ah, bl);
            bh = bh - k1 * cl;
            ah = nmul(k1,  al);
            dh = dh - k1 * dl;
            k2 = div(cl, bh);
            al = al - k2 * ah;
            cl = nmul(k2,  ch);
            dl = dl - k2 * dh;
            
            al_old = al; bl_old = bl; cl_old = cl; dl_old = dl;
            al = SHUFFLED0246(al, ah);
            bl = SHUFFLED0246(bl, bh);
            cl = SHUFFLED0246(cl, ch);
            dl = SHUFFLED0246(dl, dh);
            ah = SHUFFLED1357(al_old, ah);
            bh = SHUFFLED1357(bl_old, bh);
            ch = SHUFFLED1357(cl_old, ch);
            dh = SHUFFLED1357(dl_old, dh);
            
            /* ROUND 2 */
            all = BROADCASTD1357(al);
            bll = BROADCASTD1357(bl);
            cll = BROADCASTD1357(cl);
            dll = BROADCASTD1357(dl);
            k1  = div(ah, bll);
            e   = _mm512_maskz_mov_pd(MASK0246, bh);
            e   = e  - k1 * cll;
            ah  = nmul(k1,  all);
            dh  = dh - k1 * dll;
            bh  = _mm512_mask_mov_pd(bh, MASK0246, e);
            ahf = BROADCASTD0246(ah);
            bhf = BROADCASTD0246(e);
            chf = BROADCASTD0246(ch);
            dhf = BROADCASTD0246(dh);
            k2  = div(cl, bhf);
            k3  = maskz_div(MASKN0246, e, bhf);
            al  = al - k2 * ahf;
            cl  = nmul(k2,  chf);
            dl  = dl - k2 * dhf;
            ah  = ah - k3 * ahf;
            ch  = ch - k3 * chf;
            dh  = dh - k3 * dhf;

            al_old = al; bl_old = bl; cl_old = cl; dl_old = dl;
            al = SHUFFLE2D0123(al, ah);
            bl = SHUFFLE2D0123(bl, bh);
            cl = SHUFFLE2D0123(cl, ch);
            dl = SHUFFLE2D0123(dl, dh);
            ah = SHUFFLE2D4567(al_old, ah);
            bh = SHUFFLE2D4567(bl_old, bh);
            ch = SHUFFLE2D4567(cl_old, ch);
            dh = SHUFFLE2D4567(dl_old, dh);

            /* parallel Gaussian Elimination */
            k1  = al;
            e   = _mm512_maskz_mov_pd(MASK0, bl);
            e   = e - k1 * last_c;
            dl  = _mm512_mask3_fnmadd_pd(k1, last_d, dl, MASK0123);
            bl  = _mm512_mask_mov_pd(bl, MASK0, e);
            bhf = BROADCASTD04(e);
            chf = BROADCASTD04(cl);
            dhf = BROADCASTD04(dl);
            k3  = maskz_div(MASK123, e, bhf);
            cl  = _mm512_mask3_fnmadd_pd(k3, chf, cl, MASK123);
            dl  = _mm512_mask3_fnmadd_pd(k3, dhf, dl, MASK123);
            last_c = BROADCASTD3(cl);
            last_d = BROADCASTD3(dl);

            
            k1  = div(al, BROADCASTD3(bl));
            e   = _mm512_maskz_mov_pd(MASK4, bl);
            e   = e - k1 * last_c;
            dl  = _mm512_mask3_fnmadd_pd(k1, last_d, dl, MASK4567);
            bl  = _mm512_mask_mov_pd(bl, MASK4, e);
            bhf = BROADCASTD04(e);
            chf = BROADCASTD04(cl);
            dhf = BROADCASTD04(dl);
            k3  = maskz_div(MASK567, e, bhf);
            cl  = _mm512_mask3_fnmadd_pd(k3, chf, cl, MASK567);
            dl  = _mm512_mask3_fnmadd_pd(k3, dhf, dl, MASK567);
            cl  = div(cl, bl);
            dl  = div(dl, bl);
            _mm512_store_pd(c_mem + i, cl);
            _mm512_store_pd(d_mem + i, dl);
            last_c = BROADCASTD7(cl);
            last_d = BROADCASTD7(dl);

            k1  = ah;
            e   = _mm512_maskz_mov_pd(MASK0, bh);
            e   = e - k1 * last_c;
            dh  = _mm512_mask3_fnmadd_pd(k1, last_d, dh, MASK0123);
            bh  = _mm512_mask_mov_pd(bh, MASK0, e);
            bhf = BROADCASTD04(e);
            chf = BROADCASTD04(ch);
            dhf = BROADCASTD04(dh);
            k3  = maskz_div(MASK123, e, bhf);
            ch  = _mm512_mask3_fnmadd_pd(k3, chf, ch, MASK123);
            dh  = _mm512_mask3_fnmadd_pd(k3, dhf, dh, MASK123);
            last_c = BROADCASTD3(ch);
            last_d = BROADCASTD3(dh);
            
            k1  = div(ah, BROADCASTD3(bh));
            e   = _mm512_maskz_mov_pd(MASK4, bh);
            e   = e - k1 * last_c;
            dh  = _mm512_mask3_fnmadd_pd(k1, last_d, dh, MASK4567);
            bh  = _mm512_mask_mov_pd(bh, MASK4, e);
            bhf = BROADCASTD04(e);
            chf = BROADCASTD04(ch);
            dhf = BROADCASTD04(dh);
            k3  = maskz_div(MASK567, e, bhf);
            ch  = _mm512_mask3_fnmadd_pd(k3, chf, ch, MASK567);
            dh  = _mm512_mask3_fnmadd_pd(k3, dhf, dh, MASK567);
            ch  = div(ch, bh);
            dh  = div(dh, bh);
            _mm512_store_pd(c_mem + i + 8, ch);
            _mm512_store_pd(d_mem + i + 8, dh);
            last_c = BROADCASTD7(ch);
            last_d = BROADCASTD7(dh);
        }
        __m512d last_x = _mm512_setzero_pd();
        for(int i = n_eqt - VECLEN; i >= 0; i -= VECLEN) {
            __m512d c, d;
            c = _mm512_load_pd(c_mem + i);
            d = _mm512_load_pd(d_mem + i);
            d = _mm512_mask3_fnmadd_pd(c, last_x, d, MASK4567);
            last_x = BROADCASTD4(d);
            d = _mm512_mask3_fnmadd_pd(c, last_x, d, MASK0123);
            _mm512_store_pd(d_mem + i, d);
            last_x = BROADCASTD0(d);
        }
    PARALLEL_FOR_END
}


void PCR3_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, double *buff, int n_eqt, int n_batch) {
    PARALLEL_FOR_BEGIN(j, 0, n_batch)
        double *a_mem = a_gbl + j * n_eqt;
        double *b_mem = b_gbl + j * n_eqt;
        double *c_mem = c_gbl + j * n_eqt;
        double *d_mem = d_gbl + j * n_eqt;
        __m512d ar0, br0, cr0, dr0;
        __m512d ar1, br1, cr1, dr1;
        __m512d ar2, br2, cr2, dr2;
        __m512d last_c, last_d;
        ar0 = _mm512_setzero_pd();
        ar1 = _mm512_setzero_pd();
        ar2 = _mm512_setzero_pd();
        br0 = _mm512_set1_pd(1.0);
        br1 = _mm512_set1_pd(1.0);
        br2 = _mm512_set1_pd(1.0);
        cr0 = _mm512_setzero_pd();
        cr1 = _mm512_setzero_pd();
        cr2 = _mm512_setzero_pd();
        dr0 = _mm512_setzero_pd();
        dr1 = _mm512_setzero_pd();
        dr2 = _mm512_setzero_pd();
        last_c = _mm512_setzero_pd();
        last_d = _mm512_setzero_pd();

        for (int i = 0; i < n_eqt + VECLEN; i += VECLEN) {
            __m512d al, am, ah, arx, at;
            __m512d bl, bm, bh, brx, bt;
            __m512d cl, cm, ch, crx, ct;
            __m512d dl, dm, dh, drx, dt;
            __m512d k1, k2, kk;

            if (i < n_eqt) {
                arx = _mm512_load_pd(a_mem + i);
                brx = _mm512_load_pd(b_mem + i);
                crx = _mm512_load_pd(c_mem + i);
                drx = _mm512_load_pd(d_mem + i);
            } else {
                arx = _mm512_setzero_pd();
                brx = _mm512_set1_pd(1.0);
                crx = _mm512_setzero_pd();
                drx = _mm512_setzero_pd();
            }

            /* ROUND 1 */
            al = CONCATENATE(ar0, arx, 5);
            bl = CONCATENATE(br0, brx, 5);
            cl = CONCATENATE(cr0, crx, 5);
            dl = CONCATENATE(dr0, drx, 5);
            am = CONCATENATE(ar0, arx, 6);
            bm = CONCATENATE(br0, brx, 6);
            cm = CONCATENATE(cr0, crx, 6);
            dm = CONCATENATE(dr0, drx, 6);
            ah = CONCATENATE(ar0, arx, 7);
            bh = CONCATENATE(br0, brx, 7);
            ch = CONCATENATE(cr0, crx, 7);
            dh = CONCATENATE(dr0, drx, 7);
            ar0 = arx; br0 = brx; cr0 = crx; dr0 = drx;
            
            k1 = div(am, bl);
            k2 = div(cm, bh);
            brx = bm - k1 * cl - k2 * ah;
            drx = dm - k1 * dl - k2 * dh;
            arx = nmul(k1,  al);
            crx = nmul(          k2,  ch);

            /* ROUND 2 */
            al = CONCATENATE(ar1, arx, 4);
            bl = CONCATENATE(br1, brx, 4);
            cl = CONCATENATE(cr1, crx, 4);
            dl = CONCATENATE(dr1, drx, 4);
            am = CONCATENATE(ar1, arx, 6);
            bm = CONCATENATE(br1, brx, 6);
            cm = CONCATENATE(cr1, crx, 6);
            dm = CONCATENATE(dr1, drx, 6);
            ar1 = arx; br1 = brx; cr1 = crx; dr1 = drx;

            k1 = div(am, bl);
            k2 = div(cm, brx);
            brx = bm - k1 * cl - k2 * arx;
            drx = dm - k1 * dl - k2 * drx;
            arx = nmul(k1,  al);
            crx = nmul(          k2,  crx);

            if (i != 0) {
                /* ROUND 3 */
                am = CONCATENATE(ar2, arx, 4);
                bm = CONCATENATE(br2, brx, 4);
                cm = CONCATENATE(cr2, crx, 4);
                dm = CONCATENATE(dr2, drx, 4);

                k1 = div(am, br2);
                k2 = div(cm, brx);
                bt = bm - k1 * cr2 - k2 * arx;
                dt = dm - k1 * dr2 - k2 * drx;
                at = nmul(k1,  ar2);
                ct = nmul(          k2,  crx);

                /* parallel Thomas */
                kk = bt - last_c * at;
                last_c = div(ct              , kk);
                last_d = div(dt - last_d * at, kk);
                _mm512_store_pd(c_mem + i - VECLEN, last_c);
                _mm512_store_pd(d_mem + i - VECLEN, last_d);
            }
            ar2 = arx; br2 = brx; cr2 = crx; dr2 = drx;
        }
        __m512d x = _mm512_load_pd(d_mem + n_eqt - VECLEN);
        for (int i = n_eqt - 2 * VECLEN; i >= 0; i -= VECLEN) {
            __m512d d = _mm512_load_pd(d_mem + i);
            __m512d c = _mm512_load_pd(c_mem + i);
            x = d - c * x;
            _mm512_store_pd(d_mem + i, x);
        }
    PARALLEL_FOR_END
}

void PCR2_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, double *buff, int n_eqt, int n_batch) {
    PARALLEL_FOR_BEGIN(j, 0, n_batch)
        double *a_mem = a_gbl + j * n_eqt;
        double *b_mem = b_gbl + j * n_eqt;
        double *c_mem = c_gbl + j * n_eqt;
        double *d_mem = d_gbl + j * n_eqt;
        __m512d ar0, br0, cr0, dr0;
        __m512d ar1, br1, cr1, dr1;
        __m512d last_c, last_d;
        ar0 = _mm512_setzero_pd();
        ar1 = _mm512_setzero_pd();
        br0 = _mm512_set1_pd(1.0);
        br1 = _mm512_set1_pd(1.0);
        cr0 = _mm512_setzero_pd();
        cr1 = _mm512_setzero_pd();
        dr0 = _mm512_setzero_pd();
        dr1 = _mm512_setzero_pd();
        last_c = _mm512_setzero_pd();
        last_d = _mm512_setzero_pd();

        for (int i = 0; i < n_eqt + VECLEN; i += VECLEN) {
            __m512d al, am, ah, arx, at;
            __m512d bl, bm, bh, brx, bt;
            __m512d cl, cm, ch, crx, ct, cc0, cc1;
            __m512d dl, dm, dh, drx, dt, dd0, dd1;
            __m512d k1, k2, kk;

            if (i < n_eqt) {
                arx = _mm512_load_pd(a_mem + i);
                brx = _mm512_load_pd(b_mem + i);
                crx = _mm512_load_pd(c_mem + i);
                drx = _mm512_load_pd(d_mem + i);
            } else {
                arx = _mm512_setzero_pd();
                brx = _mm512_set1_pd(1.0);
                crx = _mm512_setzero_pd();
                drx = _mm512_setzero_pd();
            }

            /* ROUND 1 */
            al = CONCATENATE(ar0, arx, 5);
            bl = CONCATENATE(br0, brx, 5);
            cl = CONCATENATE(cr0, crx, 5);
            dl = CONCATENATE(dr0, drx, 5);
            am = CONCATENATE(ar0, arx, 6);
            bm = CONCATENATE(br0, brx, 6);
            cm = CONCATENATE(cr0, crx, 6);
            dm = CONCATENATE(dr0, drx, 6);
            ah = CONCATENATE(ar0, arx, 7);
            bh = CONCATENATE(br0, brx, 7);
            ch = CONCATENATE(cr0, crx, 7);
            dh = CONCATENATE(dr0, drx, 7);
            ar0 = arx; br0 = brx; cr0 = crx; dr0 = drx;
            
            k1 = div(am, bl);
            k2 = div(cm, bh);
            brx = bm - k1 * cl - k2 * ah;
            drx = dm - k1 * dl - k2 * dh;
            arx = nmul(k1,  al);
            crx = nmul(          k2,  ch);

            if (i != 0) {
                /* ROUND 2 */
                am = CONCATENATE(ar1, arx, 2);
                bm = CONCATENATE(br1, brx, 2);
                cm = CONCATENATE(cr1, crx, 2);
                dm = CONCATENATE(dr1, drx, 2);
                ah = CONCATENATE(ar1, arx, 4);
                bh = CONCATENATE(br1, brx, 4);
                ch = CONCATENATE(cr1, crx, 4);
                dh = CONCATENATE(dr1, drx, 4);

                k1 = div(am, br1);
                k2 = div(cm, bh);
                bt = bm - k1 * cr1 - k2 * ah;
                dt = dm - k1 * dr1 - k2 * dh;
                at = nmul(k1,  ar1);
                ct = nmul(          k2,  ch);

                /* parallel Thomas */
                kk = bt - last_c * at;
                cc0 = div(ct              , kk);
                dd0 = div(dt - last_d * at, kk);
                cc1 = CYCLICSHIFT(cc0, 4);
                dd1 = CYCLICSHIFT(dd0, 4);
                kk = bt - cc1 * at;
                cc1 = div(ct           , kk);
                dd1 = div(dt - dd1 * at, kk);
                cc0 = _mm512_mask_mov_pd(cc0, MASK4567, cc1);
                dd0 = _mm512_mask_mov_pd(dd0, MASK4567, dd1);
                _mm512_store_pd(c_mem + i - VECLEN, cc0);
                _mm512_store_pd(d_mem + i - VECLEN, dd0);
                last_c = CYCLICSHIFT(cc0, 4);
                last_d = CYCLICSHIFT(dd0, 4);
            }
            ar1 = arx; br1 = brx; cr1 = crx; dr1 = drx;
        }
        __m512d x = _mm512_setzero_pd();
        for (int i = n_eqt - VECLEN; i >= 0; i -= VECLEN) {
            __m512d c, d, xt;
            d = _mm512_load_pd(d_mem + i);
            c = _mm512_load_pd(c_mem + i);
            x = d - c * x;
            xt = CYCLICSHIFT(x, 4);
            xt = d - c * xt;
            x = _mm512_mask_mov_pd(x, MASK0123, xt);
            _mm512_store_pd(d_mem + i, x);
            x = CYCLICSHIFT(x, 4);
        }
    PARALLEL_FOR_END
}
