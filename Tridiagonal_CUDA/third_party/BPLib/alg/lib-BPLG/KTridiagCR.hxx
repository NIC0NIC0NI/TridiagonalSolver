//- =======================================================================
//+ BPLG Tridiagonal CR kernels 
//- =======================================================================

#pragma once
#ifndef KTRIDIAG_CR_HXX
#define KTRIDIAG_CR_HXX

//---- Header Dependencies -----------------------------------------------

#include "tools/cuvector.hxx"

//---- Function Declaration ----------------------------------------------


int KTridiagCR(float* data, int dir, int N, int M, int batch);


#endif // KTRIDIAG_CR_HXX

