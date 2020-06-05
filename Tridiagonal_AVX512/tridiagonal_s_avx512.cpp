#include "tridiagonal.h"
#include "threading.h"
#include "vec.h"

#define VECLEN 16

#define MASK014589CD  0x3333
#define MASK2367ABEF  0xcccc
#define MASK012389AB  0x0f0f
#define MASK4567CDEF  0xf0f0
#define MASK01234567  0x00ff
#define MASK89ABCDEF  0xff00
#define MASK02468ACE  0x5555
#define MASK13579BDF  0xaaaa //_mm512_knot(MASK02468ACE)
#define MASKN02468ACE 0xaaaa //_mm512_knot(MASK02468ACE)
#define MASK048C      0x1111
#define MASKN048C     0xeeee //_mm512_knot(MASK048C)
#define MASK08        0x0101
#define MASKN08       0xfefe //_mm512_knot(MASK08)
#define MASK0         0x0001
#define MASKN0        0xfffe //_mm512_knot(MASK08)
#define MASK8         0x0100
#define MASK1234567   0x00fe
#define MASK9ABCDEF   0xfe00
#define MASKCDEF      0xf000
#define MASKNCDEF     0x0fff
#define MASK01234     0x001f
#define MASKN01234    0xffe0

#define F2I(x)   _mm512_castps_si512(x)
#define I2F(x)   _mm512_castsi512_ps(x)

/* x[offset], ..., x[7], y[0], ..., y[offset-1] */
#define CONCATENATE(x, y, offset) I2F(_mm512_alignr_epi32(F2I(y), F2I(x), offset))
/* Shift down */
#define CYCLICSHIFT(x, offset)    I2F(_mm512_alignr_epi32(F2I(x), F2I(x), offset))

#define BROADCASTS02468ACE(x)  _mm512_moveldup_ps(x)
#define BROADCASTS13579BDF(x)  _mm512_movehdup_ps(x)
#define BROADCASTS048C(x)      _mm512_permute_ps(x, 0x00)
#define BROADCASTS37BE(x)      _mm512_permute_ps(x, 0xff)
#define BROADCASTS08(x)        _mm512_permutexvar_ps(idx08bcast, x)
#define BROADCASTS7F(x)        _mm512_permutexvar_ps(idx7Fbcast, x) 
#define BROADCASTS0(x)         _mm512_broadcastss_ps(_mm512_castps512_ps128(x))
#define BROADCASTS7(x)         _mm512_permutexvar_ps(idx7bcast, x)
#define BROADCASTS8(x)         _mm512_permutexvar_ps(idx8bcast, x)
#define BROADCASTSF(x)         _mm512_permutexvar_ps(idxFbcast, x)

/* READ to ROUND1 */
#define PACKS02468ACE(x, y)    _mm512_permutex2var_ps(x, idxl, y)
#define PACKS13579BDF(x, y)    _mm512_permutex2var_ps(x, idxh, y)

/* ROUND1 to ROUND2 */
#define SHUFFLES02468ACE(x, y) _mm512_mask_moveldup_ps(x, MASK13579BDF, y)
#define SHUFFLES13579BDF(x, y) _mm512_mask_movehdup_ps(y, MASK02468ACE, x)

/* ROUND2 to ROUND3 */
#ifdef USE_SHUFFLE
#define SHUFFLE2S014589CD(x, y) _mm512_shuffle_ps(x, y, 0x44)
#define SHUFFLE2S2367ABEF(x, y) _mm512_shuffle_ps(x, y, 0xee)
#else
/* valignd is faster than vshufps and vshuff32x4, but uses an additional mask */
#define SHUFFLE2S014589CD(x, y) I2F(_mm512_mask_alignr_epi32(F2I(x), MASK2367ABEF, F2I(y), F2I(y), 14))
#define SHUFFLE2S2367ABEF(x, y) I2F(_mm512_mask_alignr_epi32(F2I(y), MASK014589CD, F2I(x), F2I(x), 2))
#endif

/* ROUND3 to ROUND4 */
#define SHUFFLE4S012389AB(x, y) I2F(_mm512_mask_alignr_epi32(F2I(x), MASK4567CDEF, F2I(y), F2I(y), 12))
#define SHUFFLE4S4567CDEF(x, y) I2F(_mm512_mask_alignr_epi32(F2I(y), MASK012389AB, F2I(x), F2I(x), 4))

/* ROUND4 to pGE */
#ifdef USE_SHUFFLE
#define PACKS01234567(x, y)     _mm512_shuffle_f32x4(x, y, 0x44)
#define PACKS89ABCDEF(x, y)     _mm512_shuffle_f32x4(x, y, 0xee)
#else
#define PACKS01234567(x, y)     I2F(_mm512_mask_alignr_epi32(F2I(x), MASK89ABCDEF, F2I(y), F2I(y), 8))
#define PACKS89ABCDEF(x, y)     I2F(_mm512_mask_alignr_epi32(F2I(y), MASK01234567, F2I(x), F2I(x), 8))
#endif

/* ROUND3 to pGE */
#define SHUFFLE4S01234567(x, y) _mm512_permutex2var_ps(x, idx16l, y)
#define SHUFFLE4S89ABCDEF(x, y) _mm512_permutex2var_ps(x, idx16h, y)

#define COMPILER_FENCE(x) __asm__ __volatile__ ("" :"+v"(x))

void WM4_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, float *buf, int n_eqt, int n_batch) {
    PARALLEL_FOR_BEGIN(j, 0, n_batch)
        const __m512i idxFbcast = _mm512_set1_epi32(15);
        const __m512i idx08bcast = _mm512_set_epi32(8,8,8,8,8,8,8,8,0,0,0,0,0,0,0,0);
        const __m512i idx7Fbcast = _mm512_set_epi32(15,15,15,15,15,15,15,15,7,7,7,7,7,7,7,7);
        const __m512i idxl = _mm512_set_epi32(30,28,26,24,22,20,18,16,14,12,10,8,6,4,2,0);
        const __m512i idxh = _mm512_set_epi32(31,29,27,25,23,21,19,17,15,13,11,9,7,5,3,1);
        float *a_mem = a_gbl + j * n_eqt;
        float *b_mem = b_gbl + j * n_eqt;
        float *c_mem = c_gbl + j * n_eqt;
        float *d_mem = d_gbl + j * n_eqt;
        __m512 last_c = _mm512_setzero_ps(), last_d = _mm512_setzero_ps();
        for (int i = 0; i < n_eqt; i += VECLEN * 2) {
            /* l: low, h: high, ll: low last, hf: high first */
            __m512 al, ah, al_old, all, ahf;
            __m512 bl, bh, bl_old, bll, bhf;
            __m512 cl, ch, cl_old, cll, chf;
            __m512 dl, dh, dl_old, dll, dhf;
            __m512 k1, k2, k3, e;

            al = _mm512_load_ps(a_mem + i);
            bl = _mm512_load_ps(b_mem + i);
            cl = _mm512_load_ps(c_mem + i);
            dl = _mm512_load_ps(d_mem + i);
            ah = _mm512_load_ps(a_mem + i + VECLEN);
            bh = _mm512_load_ps(b_mem + i + VECLEN);
            ch = _mm512_load_ps(c_mem + i + VECLEN);
            dh = _mm512_load_ps(d_mem + i + VECLEN);
           
            al_old = al; bl_old = bl; cl_old = cl; dl_old = dl;
            al = PACKS02468ACE(al, ah);
            bl = PACKS02468ACE(bl, bh);
            cl = PACKS02468ACE(cl, ch);
            dl = PACKS02468ACE(dl, dh);
            ah = PACKS13579BDF(al_old, ah);
            bh = PACKS13579BDF(bl_old, bh);
            ch = PACKS13579BDF(cl_old, ch);
            dh = PACKS13579BDF(dl_old, dh);

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
            al = SHUFFLES02468ACE(al, ah);
            bl = SHUFFLES02468ACE(bl, bh);
            cl = SHUFFLES02468ACE(cl, ch);
            dl = SHUFFLES02468ACE(dl, dh);
            ah = SHUFFLES13579BDF(al_old, ah);
            bh = SHUFFLES13579BDF(bl_old, bh);
            ch = SHUFFLES13579BDF(cl_old, ch);
            dh = SHUFFLES13579BDF(dl_old, dh);

            /* ROUND 2 */
            all = BROADCASTS13579BDF(al);
            bll = BROADCASTS13579BDF(bl);
            cll = BROADCASTS13579BDF(cl);
            dll = BROADCASTS13579BDF(dl);
            k1  = div(ah, bll);
            e   = _mm512_maskz_mov_ps(MASK02468ACE, bh);
            e   = e  - k1 * cll;
            ah  = nmul(k1,  all);
            dh  = dh - k1 * dll;
            bh  = _mm512_mask_mov_ps(bh, MASK02468ACE, e);
            ahf = BROADCASTS02468ACE(ah);
            bhf = BROADCASTS02468ACE(e);
            chf = BROADCASTS02468ACE(ch);
            dhf = BROADCASTS02468ACE(dh);
            k2  = div(cl, bhf);
            k3  = maskz_div(MASKN02468ACE, e, bhf);
            al  = al - k2 * ahf;
            cl  = nmul(k2,  chf);
            dl  = dl - k2 * dhf;
            ah  = ah - k3 * ahf;
            ch  = ch - k3 * chf;
            dh  = dh - k3 * dhf;

            al_old = al; bl_old = bl; cl_old = cl; dl_old = dl;
            al = SHUFFLE2S014589CD(al, ah);
            bl = SHUFFLE2S014589CD(bl, bh);
            cl = SHUFFLE2S014589CD(cl, ch);
            dl = SHUFFLE2S014589CD(dl, dh);
            ah = SHUFFLE2S2367ABEF(al_old, ah);
            bh = SHUFFLE2S2367ABEF(bl_old, bh);
            ch = SHUFFLE2S2367ABEF(cl_old, ch);
            dh = SHUFFLE2S2367ABEF(dl_old, dh);

            /* ROUND 3 */
            all = BROADCASTS37BE(al);
            bll = BROADCASTS37BE(bl);
            cll = BROADCASTS37BE(cl);
            dll = BROADCASTS37BE(dl);
            k1  = div(ah, bll);
            e   = _mm512_maskz_mov_ps(MASK048C, bh);
            e   = e  - k1 * cll;
            ah  = nmul(k1,  all);
            dh  = dh - k1 * dll;
            bh  = _mm512_mask_mov_ps(bh, MASK048C, e);
            ahf = BROADCASTS048C(ah);
            bhf = BROADCASTS048C(e);
            chf = BROADCASTS048C(ch);
            dhf = BROADCASTS048C(dh);
            k2  = div(cl, bhf);
            k3  = maskz_div(MASKN048C, e, bhf);
            al  = al - k2 * ahf;
            cl  = nmul(k2,  chf);
            dl  = dl - k2 * dhf;
            ah  = ah - k3 * ahf;
            ch  = ch - k3 * chf;
            dh  = dh - k3 * dhf;

            al_old = al; bl_old = bl; cl_old = cl; dl_old = dl;
            al = SHUFFLE4S012389AB(al, ah);
            bl = SHUFFLE4S012389AB(bl, bh);
            cl = SHUFFLE4S012389AB(cl, ch);
            dl = SHUFFLE4S012389AB(dl, dh);
            ah = SHUFFLE4S4567CDEF(al_old, ah);
            bh = SHUFFLE4S4567CDEF(bl_old, bh);
            ch = SHUFFLE4S4567CDEF(cl_old, ch);
            dh = SHUFFLE4S4567CDEF(dl_old, dh);

            /* ROUND 4 */
            all = BROADCASTS7F(al);
            bll = BROADCASTS7F(bl);
            cll = BROADCASTS7F(cl);
            dll = BROADCASTS7F(dl);
            k1  = div(ah, bll);
            e   = _mm512_maskz_mov_ps(MASK08, bh);
            e   = e  - k1 * cll;
            ah  = nmul(k1,  all);
            dh  = dh - k1 * dll;
            bh  = _mm512_mask_mov_ps(bh, MASK08, e);
            ahf = BROADCASTS08(ah);
            bhf = BROADCASTS08(e);
            chf = BROADCASTS08(ch);
            dhf = BROADCASTS08(dh);
            k2  = div(cl, bhf);
            k3  = maskz_div(MASKN08, e, bhf);
            al  = al - k2 * ahf;
            cl  = nmul(k2,  chf);
            dl  = dl - k2 * dhf;
            ah  = ah - k3 * ahf;
            ch  = ch - k3 * chf;
            dh  = dh - k3 * dhf;

            al_old = al; bl_old = bl; cl_old = cl; dl_old = dl;
            al = PACKS01234567(al, ah);
            bl = PACKS01234567(bl, bh);
            cl = PACKS01234567(cl, ch);
            dl = PACKS01234567(dl, dh);
            ah = PACKS89ABCDEF(al_old, ah);
            bh = PACKS89ABCDEF(bl_old, bh);
            ch = PACKS89ABCDEF(cl_old, ch);
            dh = PACKS89ABCDEF(dl_old, dh);

            /* parallel Gaussian Elimination */
            e   = _mm512_maskz_mov_ps(MASK0, bl);
            e   = e  - al * last_c;
            dl  = dl - al * last_d;
            bl  = _mm512_mask_mov_ps(bl, MASK0, e);
            bhf = BROADCASTS0(e);
            chf = BROADCASTS0(cl);
            dhf = BROADCASTS0(dl);
            k3  = maskz_div(MASKN0, e, bhf);
            cl  = div(cl - k3 * chf, bl);
            dl  = div(dl - k3 * dhf, bl);
            _mm512_store_ps(c_mem + i, cl);
            _mm512_store_ps(d_mem + i, dl);
            last_c = BROADCASTSF(cl);
            last_d = BROADCASTSF(dl);

            e   = _mm512_maskz_mov_ps(MASK0, bh);
            e   = e  - ah * last_c;
            dh  = dh - ah * last_d;
            bh  = _mm512_mask_mov_ps(bh, MASK0, e);
            bhf = BROADCASTS0(e);
            chf = BROADCASTS0(ch);
            dhf = BROADCASTS0(dh);
            k3  = maskz_div(MASKN0, e, bhf);
            ch  = div(ch - k3 * chf, bh);
            dh  = div(dh - k3 * dhf, bh);
            _mm512_store_ps(c_mem + i + VECLEN, ch);
            _mm512_store_ps(d_mem + i + VECLEN, dh);
            last_c = BROADCASTSF(ch);
            last_d = BROADCASTSF(dh);
        }
        __m512 last_x = BROADCASTS0(_mm512_load_ps(d_mem + n_eqt - VECLEN));
        for(int i = n_eqt - VECLEN * 2; i >= 0; i -= VECLEN) {
            __m512 c, d, x;
            c = _mm512_load_ps(c_mem + i);
            d = _mm512_load_ps(d_mem + i);
            x = d - c * last_x;
            _mm512_store_ps(d_mem + i, x);
            last_x = BROADCASTS0(x);
        }
    PARALLEL_FOR_END
}

void WM3_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, float *buf, int n_eqt, int n_batch) {
    PARALLEL_FOR_BEGIN(j, 0, n_batch)
        const __m512i idx7bcast = _mm512_set1_epi32(7);
        const __m512i idx8bcast = _mm512_set1_epi32(8);
        const __m512i idxFbcast = _mm512_set1_epi32(15);
        const __m512i idx08bcast = _mm512_set_epi32(8,8,8,8,8,8,8,8,0,0,0,0,0,0,0,0);
        const __m512i idx7Fbcast = _mm512_set_epi32(15,15,15,15,15,15,15,15,7,7,7,7,7,7,7,7);
        const __m512i idxl = _mm512_set_epi32(30,28,26,24,22,20,18,16,14,12,10,8,6,4,2,0);
        const __m512i idxh = _mm512_set_epi32(31,29,27,25,23,21,19,17,15,13,11,9,7,5,3,1);
        const __m512i idx16l = _mm512_set_epi32(23,22,21,20,7,6,5,4,19,18,17,16,3,2,1,0);
        const __m512i idx16h = _mm512_set_epi32(31,30,29,28,15,14,13,12,27,26,25,24,11,10,9,8);
        float *a_mem = a_gbl + j * n_eqt;
        float *b_mem = b_gbl + j * n_eqt;
        float *c_mem = c_gbl + j * n_eqt;
        float *d_mem = d_gbl + j * n_eqt;
        __m512 last_c = _mm512_setzero_ps(), last_d = _mm512_setzero_ps();
        for (int i = 0; i < n_eqt; i += VECLEN * 2) {
            /* l: low, h: high, ll: low last, hf: high first */
            __m512 al, ah, al_old, all, ahf;
            __m512 bl, bh, bl_old, bll, bhf;
            __m512 cl, ch, cl_old, cll, chf;
            __m512 dl, dh, dl_old, dll, dhf;
            __m512 k1, k2, k3, e;

            al = _mm512_load_ps(a_mem + i);
            bl = _mm512_load_ps(b_mem + i);
            cl = _mm512_load_ps(c_mem + i);
            dl = _mm512_load_ps(d_mem + i);
            ah = _mm512_load_ps(a_mem + i + VECLEN);
            bh = _mm512_load_ps(b_mem + i + VECLEN);
            ch = _mm512_load_ps(c_mem + i + VECLEN);
            dh = _mm512_load_ps(d_mem + i + VECLEN);
            
            al_old = al; bl_old = bl; cl_old = cl; dl_old = dl;
            al = PACKS02468ACE(al, ah);
            bl = PACKS02468ACE(bl, bh);
            cl = PACKS02468ACE(cl, ch);
            dl = PACKS02468ACE(dl, dh);
            ah = PACKS13579BDF(al_old, ah);
            bh = PACKS13579BDF(bl_old, bh);
            ch = PACKS13579BDF(cl_old, ch);
            dh = PACKS13579BDF(dl_old, dh);

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
            al = SHUFFLES02468ACE(al, ah);
            bl = SHUFFLES02468ACE(bl, bh);
            cl = SHUFFLES02468ACE(cl, ch);
            dl = SHUFFLES02468ACE(dl, dh);
            ah = SHUFFLES13579BDF(al_old, ah);
            bh = SHUFFLES13579BDF(bl_old, bh);
            ch = SHUFFLES13579BDF(cl_old, ch);
            dh = SHUFFLES13579BDF(dl_old, dh);

            /* ROUND 2 */
            all = BROADCASTS13579BDF(al);
            bll = BROADCASTS13579BDF(bl);
            cll = BROADCASTS13579BDF(cl);
            dll = BROADCASTS13579BDF(dl);
            k1  = div(ah, bll);
            e   = _mm512_maskz_mov_ps(MASK02468ACE, bh);
            e   = e  - k1 * cll;
            ah  = nmul(k1,  all);
            dh  = dh - k1 * dll;
            bh  = _mm512_mask_mov_ps(bh, MASK02468ACE, e);
            ahf = BROADCASTS02468ACE(ah);
            bhf = BROADCASTS02468ACE(e);
            chf = BROADCASTS02468ACE(ch);
            dhf = BROADCASTS02468ACE(dh);
            k2  = div(cl, bhf);
            k3  = maskz_div(MASKN02468ACE, e, bhf);
            al  = al - k2 * ahf;
            cl  = nmul(k2,  chf);
            dl  = dl - k2 * dhf;
            ah  = ah - k3 * ahf;
            ch  = ch - k3 * chf;
            dh  = dh - k3 * dhf;

            al_old = al; bl_old = bl; cl_old = cl; dl_old = dl;
            al = SHUFFLE2S014589CD(al, ah);
            bl = SHUFFLE2S014589CD(bl, bh);
            cl = SHUFFLE2S014589CD(cl, ch);
            dl = SHUFFLE2S014589CD(dl, dh);
            ah = SHUFFLE2S2367ABEF(al_old, ah);
            bh = SHUFFLE2S2367ABEF(bl_old, bh);
            ch = SHUFFLE2S2367ABEF(cl_old, ch);
            dh = SHUFFLE2S2367ABEF(dl_old, dh);

            /* ROUND 3 */
            all = BROADCASTS37BE(al);
            bll = BROADCASTS37BE(bl);
            cll = BROADCASTS37BE(cl);
            dll = BROADCASTS37BE(dl);
            k1  = div(ah, bll);
            e   = _mm512_maskz_mov_ps(MASK048C, bh);
            e   = e  - k1 * cll;
            ah  = nmul(k1,  all);
            dh  = dh - k1 * dll;
            bh  = _mm512_mask_mov_ps(bh, MASK048C, e);
            ahf = BROADCASTS048C(ah);
            bhf = BROADCASTS048C(e);
            chf = BROADCASTS048C(ch);
            dhf = BROADCASTS048C(dh);
            k2  = div(cl, bhf);
            k3  = maskz_div(MASKN048C, e, bhf);
            al  = al - k2 * ahf;
            cl  = nmul(k2,  chf);
            dl  = dl - k2 * dhf;
            ah  = ah - k3 * ahf;
            ch  = ch - k3 * chf;
            dh  = dh - k3 * dhf;

            al_old = al; bl_old = bl; cl_old = cl; dl_old = dl;
            al = SHUFFLE4S01234567(al, ah);
            bl = SHUFFLE4S01234567(bl, bh);
            cl = SHUFFLE4S01234567(cl, ch);
            dl = SHUFFLE4S01234567(dl, dh);
            ah = SHUFFLE4S89ABCDEF(al_old, ah);
            bh = SHUFFLE4S89ABCDEF(bl_old, bh);
            ch = SHUFFLE4S89ABCDEF(cl_old, ch);
            dh = SHUFFLE4S89ABCDEF(dl_old, dh);

            /* parallel Gaussian Elimination */
            k1  = al;
            e   = _mm512_maskz_mov_ps(MASK0, bl);
            e   = e - k1 * last_c;
            dl  = _mm512_mask3_fnmadd_ps(k1, last_d, dl, MASK01234567);
            bl  = _mm512_mask_mov_ps(bl, MASK0, e);
            bhf = BROADCASTS08(e);
            chf = BROADCASTS08(cl);
            dhf = BROADCASTS08(dl);
            k3  = maskz_div(MASK1234567, e, bhf);
            cl  = _mm512_mask3_fnmadd_ps(k3, chf, cl, MASK1234567);
            dl  = _mm512_mask3_fnmadd_ps(k3, dhf, dl, MASK1234567);
            last_c = BROADCASTS7(cl);
            last_d = BROADCASTS7(dl);

            
            k1  = div(al, BROADCASTS7(bl));
            e   = _mm512_maskz_mov_ps(MASK8, bl);
            e   = e - k1 * last_c;
            dl  = _mm512_mask3_fnmadd_ps(k1, last_d, dl, MASK89ABCDEF);
            bl  = _mm512_mask_mov_ps(bl, MASK8, e);
            bhf = BROADCASTS08(e);
            chf = BROADCASTS08(cl);
            dhf = BROADCASTS08(dl);
            k3  = maskz_div(MASK9ABCDEF, e, bhf);
            cl  = _mm512_mask3_fnmadd_ps(k3, chf, cl, MASK9ABCDEF);
            dl  = _mm512_mask3_fnmadd_ps(k3, dhf, dl, MASK9ABCDEF);
            cl  = div(cl, bl);
            dl  = div(dl, bl);
            _mm512_store_ps(c_mem + i, cl);
            _mm512_store_ps(d_mem + i, dl);
            last_c = BROADCASTSF(cl);
            last_d = BROADCASTSF(dl);

            k1  = ah;
            e   = _mm512_maskz_mov_ps(MASK0, bh);
            e   = e - k1 * last_c;
            dh  = _mm512_mask3_fnmadd_ps(k1, last_d, dh, MASK01234567);
            bh  = _mm512_mask_mov_ps(bh, MASK0, e);
            bhf = BROADCASTS08(e);
            chf = BROADCASTS08(ch);
            dhf = BROADCASTS08(dh);
            k3  = maskz_div(MASK1234567, e, bhf);
            ch  = _mm512_mask3_fnmadd_ps(k3, chf, ch, MASK1234567);
            dh  = _mm512_mask3_fnmadd_ps(k3, dhf, dh, MASK1234567);
            last_c = BROADCASTS7(ch);
            last_d = BROADCASTS7(dh);

            
            k1  = div(ah, BROADCASTS7(bh));
            e   = _mm512_maskz_mov_ps(MASK8, bh);
            e   = e - k1 * last_c;
            dh  = _mm512_mask3_fnmadd_ps(k1, last_d, dh, MASK89ABCDEF);
            bh  = _mm512_mask_mov_ps(bh, MASK8, e);
            bhf = BROADCASTS08(e);
            chf = BROADCASTS08(ch);
            dhf = BROADCASTS08(dh);
            k3  = maskz_div(MASK9ABCDEF, e, bhf);
            ch  = _mm512_mask3_fnmadd_ps(k3, chf, ch, MASK9ABCDEF);
            dh  = _mm512_mask3_fnmadd_ps(k3, dhf, dh, MASK9ABCDEF);
            ch  = div(ch, bh);
            dh  = div(dh, bh);
            _mm512_store_ps(c_mem + i + VECLEN, ch);
            _mm512_store_ps(d_mem + i + VECLEN, dh);
            last_c = BROADCASTSF(ch);
            last_d = BROADCASTSF(dh);
        }
        __m512 last_x = _mm512_setzero_ps();
        for(int i = n_eqt - VECLEN; i >= 0; i -= VECLEN) {
            __m512 c, d;
            c = _mm512_load_ps(c_mem + i);
            d = _mm512_load_ps(d_mem + i);
            d = _mm512_mask3_fnmadd_ps(c, last_x, d, MASK89ABCDEF);
            last_x = BROADCASTS8(d);
            d = _mm512_mask3_fnmadd_ps(c, last_x, d, MASK01234567);
            _mm512_store_ps(d_mem + i, d);
            last_x = BROADCASTS0(d);
        }
    PARALLEL_FOR_END
}

void PCR4_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, float *buf, int n_eqt, int n_batch) {
    PARALLEL_FOR_BEGIN(j, 0, n_batch)
        float *a_mem = a_gbl + j * n_eqt;
        float *b_mem = b_gbl + j * n_eqt;
        float *c_mem = c_gbl + j * n_eqt;
        float *d_mem = d_gbl + j * n_eqt;
        __m512 ar01, br01, cr01, dr01; /* combined to avoid spilling */
        __m512 ar2, br2, cr2, dr2;
        __m512 ar3, br3, cr3, dr3;
        __m512 last_c, last_d;
        ar01 = _mm512_setzero_ps();
        ar2 = _mm512_setzero_ps();
        ar3 = _mm512_setzero_ps();
        br01 = _mm512_set1_ps(1.0f);
        br2 = _mm512_set1_ps(1.0f);
        br3 = _mm512_set1_ps(1.0f);
        cr01 = _mm512_setzero_ps();
        cr2 = _mm512_setzero_ps();
        cr3 = _mm512_setzero_ps();
        dr01 = _mm512_setzero_ps();
        dr2 = _mm512_setzero_ps();
        dr3 = _mm512_setzero_ps();

        for (int i = 0; i < n_eqt + VECLEN; i += VECLEN) {
            __m512 al, am, ah, arx, at, att;
            __m512 bl, bm, bh, brx, bt, btt;
            __m512 cl, cm, ch, crx, ct, ctt;
            __m512 dl, dm, dh, drx, dt, dtt;
            __m512 k1, k2, kk;

            if (i < n_eqt) {
                arx = _mm512_load_ps(a_mem + i);
                brx = _mm512_load_ps(b_mem + i);
                crx = _mm512_load_ps(c_mem + i);
                drx = _mm512_load_ps(d_mem + i);
            } else {
                arx = _mm512_setzero_ps();
                brx = _mm512_set1_ps(1.0f);
                crx = _mm512_setzero_ps();
                drx = _mm512_setzero_ps();
            }
            
            /* ROUND 1 */
            al = CONCATENATE(ar01, arx, 13);
            bl = CONCATENATE(br01, brx, 13);
            cl = CONCATENATE(cr01, crx, 13);
            dl = CONCATENATE(dr01, drx, 13);
            am = CONCATENATE(ar01, arx, 14);
            bm = CONCATENATE(br01, brx, 14);
            cm = CONCATENATE(cr01, crx, 14);
            dm = CONCATENATE(dr01, drx, 14);
            ah = CONCATENATE(ar01, arx, 15);
            bh = CONCATENATE(br01, brx, 15);
            ch = CONCATENATE(cr01, crx, 15);
            dh = CONCATENATE(dr01, drx, 15);
            
            k1 = div(am, bl);
            k2 = div(cm, bh);
            bt = bm - k1 * cl - k2 * ah;
            dt = dm - k1 * dl - k2 * dh;
            at = nmul(k1,  al);
            ct = nmul(          k2,  ch);

            /* compress r0 and r1 to avoid spilling */
            att = CYCLICSHIFT(at, 4);
            btt = CYCLICSHIFT(bt, 4);
            ctt = CYCLICSHIFT(ct, 4);
            dtt = CYCLICSHIFT(dt, 4);
            ar01 = _mm512_mask_mov_ps(ar01, MASKCDEF, att);
            br01 = _mm512_mask_mov_ps(br01, MASKCDEF, btt);
            cr01 = _mm512_mask_mov_ps(cr01, MASKCDEF, ctt);
            dr01 = _mm512_mask_mov_ps(dr01, MASKCDEF, dtt);
            arx = _mm512_mask_mov_ps(arx, MASKNCDEF, att);
            brx = _mm512_mask_mov_ps(brx, MASKNCDEF, btt);
            crx = _mm512_mask_mov_ps(crx, MASKNCDEF, ctt);
            drx = _mm512_mask_mov_ps(drx, MASKNCDEF, dtt);

            /* ROUND 2 */
            al = CONCATENATE(ar01, arx, 8);
            bl = CONCATENATE(br01, brx, 8);
            cl = CONCATENATE(cr01, crx, 8);
            dl = CONCATENATE(dr01, drx, 8);
            am = CONCATENATE(ar01, arx, 10);
            bm = CONCATENATE(br01, brx, 10);
            cm = CONCATENATE(cr01, crx, 10);
            dm = CONCATENATE(dr01, drx, 10);
            ah = at;    bh = bt;    ch = ct;    dh = dt;
            ar01 = arx; br01 = brx; cr01 = crx; dr01 = drx;

            k1 = div(am, bl); 
            k2 = div(cm, bh);
            brx = bm - k1 * cl - k2 * ah;
            drx = dm - k1 * dl - k2 * dh;
            arx = nmul(k1,  al);
            crx = nmul(          k2,  ch);

            /* ROUND 3 */
            al = CONCATENATE(ar2, arx, 8);
            bl = CONCATENATE(br2, brx, 8);
            cl = CONCATENATE(cr2, crx, 8);
            dl = CONCATENATE(dr2, drx, 8);
            am = CONCATENATE(ar2, arx, 12);
            bm = CONCATENATE(br2, brx, 12);
            cm = CONCATENATE(cr2, crx, 12);
            dm = CONCATENATE(dr2, drx, 12);
            ar2 = arx; br2 = brx; cr2 = crx; dr2 = drx;

            k1 = div(am, bl); 
            k2 = div(cm, brx);
            brx = bm - k1 * cl - k2 * arx;
            drx = dm - k1 * dl - k2 * drx;
            arx = nmul(k1,  al);
            crx = nmul(          k2,  crx);

            if (i != 0) {
                /* ROUND 4 */
                am = CONCATENATE(ar3, arx, 8);
                bm = CONCATENATE(br3, brx, 8);
                cm = CONCATENATE(cr3, crx, 8);
                dm = CONCATENATE(dr3, drx, 8);

                COMPILER_FENCE(brx); /* magic to workaround compiler bugs, don't touch */
                COMPILER_FENCE(br3); /* magic to workaround compiler bugs, don't touch */

                k1 = div(am, br3); 
                k2 = div(cm, brx);
                bt = bm - k1 * cr3 - k2 * arx;
                dt = dm - k1 * dr3 - k2 * drx;
                at = nmul(k1,  ar3);
                ct = nmul(           k2,  crx);

                /* parallel Thomas */
                kk = bt - last_c * at;
                last_c = div(ct              , kk);
                last_d = div(dt - last_d * at, kk);
                _mm512_store_ps(c_mem + i - VECLEN, last_c);
                _mm512_store_ps(d_mem + i - VECLEN, last_d);
            }
            ar3 = arx; br3 = brx; cr3 = crx; dr3 = drx;
        }
        __m512 x = _mm512_load_ps(d_mem + n_eqt - VECLEN);
        for (int i = n_eqt - 2 * VECLEN; i >= 0; i -= VECLEN) {
            __m512 d = _mm512_load_ps(d_mem + i);
            __m512 c = _mm512_load_ps(c_mem + i);
            x = d - c * x;
            _mm512_store_ps(d_mem + i, x);
        }
    PARALLEL_FOR_END
}

void PCR3_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, float *buf, int n_eqt, int n_batch) {
    PARALLEL_FOR_BEGIN(j, 0, n_batch)
        float *a_mem = a_gbl + j * n_eqt;
        float *b_mem = b_gbl + j * n_eqt;
        float *c_mem = c_gbl + j * n_eqt;
        float *d_mem = d_gbl + j * n_eqt;
        __m512 ar0, br0, cr0, dr0;
        __m512 ar1, br1, cr1, dr1;
        __m512 ar2, br2, cr2, dr2;
        __m512 last_c, last_d;
        ar0 = _mm512_setzero_ps();
        ar1 = _mm512_setzero_ps();
        ar2 = _mm512_setzero_ps();
        br0 = _mm512_set1_ps(1.0f);
        br1 = _mm512_set1_ps(1.0f);
        br2 = _mm512_set1_ps(1.0f);
        cr0 = _mm512_setzero_ps();
        cr1 = _mm512_setzero_ps();
        cr2 = _mm512_setzero_ps();
        dr0 = _mm512_setzero_ps();
        dr1 = _mm512_setzero_ps();
        dr2 = _mm512_setzero_ps();

        for (int i = 0; i < n_eqt + VECLEN; i += VECLEN) {
            __m512 al, am, ah, arx, at;
            __m512 bl, bm, bh, brx, bt;
            __m512 cl, cm, ch, crx, ct, cc0, cc1;
            __m512 dl, dm, dh, drx, dt, dd0, dd1;
            __m512 k1, k2, kk;

            if (i < n_eqt) {
                arx = _mm512_load_ps(a_mem + i);
                brx = _mm512_load_ps(b_mem + i);
                crx = _mm512_load_ps(c_mem + i);
                drx = _mm512_load_ps(d_mem + i);
            } else {
                arx = _mm512_setzero_ps();
                brx = _mm512_set1_ps(1.0f);
                crx = _mm512_setzero_ps();
                drx = _mm512_setzero_ps();
            }
            
            /* ROUND 1 */
            al = CONCATENATE(ar0, arx, 13);
            bl = CONCATENATE(br0, brx, 13);
            cl = CONCATENATE(cr0, crx, 13);
            dl = CONCATENATE(dr0, drx, 13);
            am = CONCATENATE(ar0, arx, 14);
            bm = CONCATENATE(br0, brx, 14);
            cm = CONCATENATE(cr0, crx, 14);
            dm = CONCATENATE(dr0, drx, 14);
            ah = CONCATENATE(ar0, arx, 15);
            bh = CONCATENATE(br0, brx, 15);
            ch = CONCATENATE(cr0, crx, 15);
            dh = CONCATENATE(dr0, drx, 15);
            ar0 = arx; br0 = brx; cr0 = crx; dr0 = drx;
            
            k1 = div(am, bl);
            k2 = div(cm, bh);
            brx = bm - k1 * cl - k2 * ah;
            drx = dm - k1 * dl - k2 * dh;
            arx = nmul(k1,  al);
            crx = nmul(          k2,  ch);

            /* ROUND 2 */
            al = CONCATENATE(ar1, arx, 12);
            bl = CONCATENATE(br1, brx, 12);
            cl = CONCATENATE(cr1, crx, 12);
            dl = CONCATENATE(dr1, drx, 12);
            am = CONCATENATE(ar1, arx, 14);
            bm = CONCATENATE(br1, brx, 14);
            cm = CONCATENATE(cr1, crx, 14);
            dm = CONCATENATE(dr1, drx, 14);
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
                ah = CONCATENATE(ar2, arx, 8);
                bh = CONCATENATE(br2, brx, 8);
                ch = CONCATENATE(cr2, crx, 8);
                dh = CONCATENATE(dr2, drx, 8);

                COMPILER_FENCE(bh); /* magic to workaround compiler bugs, don't touch */
                COMPILER_FENCE(br2); /* magic to workaround compiler bugs, don't touch */

                k1 = div(am, br2);
                k2 = div(cm, bh);
                bt = bm - k1 * cr2 - k2 * ah;
                dt = dm - k1 * dr2 - k2 * dh;
                at = nmul(k1,  ar2);
                ct = nmul(           k2,  ch);

                /* parallel Thomas */
                kk = bt - last_c * at;
                cc0 = div(ct              , kk);
                dd0 = div(dt - last_d * at, kk);
                cc1 = CYCLICSHIFT(cc0, 8);
                dd1 = CYCLICSHIFT(dd0, 8);
                kk = bt - cc1 * at;
                cc1 = div(ct           , kk);
                dd1 = div(dt - dd1 * at, kk);
                cc0 = _mm512_mask_mov_ps(cc0, MASK89ABCDEF, cc1);
                dd0 = _mm512_mask_mov_ps(dd0, MASK89ABCDEF, dd1);
                _mm512_store_ps(c_mem + i - VECLEN, cc0);
                _mm512_store_ps(d_mem + i - VECLEN, dd0);
                last_c = CYCLICSHIFT(cc0, 8);
                last_d = CYCLICSHIFT(dd0, 8);
            }
            ar2 = arx; br2 = brx; cr2 = crx; dr2 = drx;
        }
        __m512 x = _mm512_setzero_ps();
        for (int i = n_eqt - VECLEN; i >= 0; i -= VECLEN) {
            __m512 c, d, xt;
            d = _mm512_load_ps(d_mem + i);
            c = _mm512_load_ps(c_mem + i);
            x = d - c * x;
            xt = CYCLICSHIFT(x, 8);
            xt = d - c * xt;
            x = _mm512_mask_mov_ps(x, MASK01234567, xt);
            _mm512_store_ps(d_mem + i, x);
            x = CYCLICSHIFT(x, 8);
        }
    PARALLEL_FOR_END
}

