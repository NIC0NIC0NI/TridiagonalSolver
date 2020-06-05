//- =======================================================================
//+ BPLG Tridiagonal WM kernels 
//- =======================================================================

#pragma once
#ifndef KTRIDIAG_WM_HXX
#define KTRIDIAG_WM_HXX

//---- Header Dependencies -----------------------------------------------

#include "tools/cuvector.hxx"

//---- Function Declaration ----------------------------------------------


int KTridiagWM(float* data, int dir, int N, int M, int batch);


#endif // KTRIDIAG_PCR_WM

