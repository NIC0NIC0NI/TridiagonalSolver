//- =======================================================================
//+ BPLG Tridiagonal kernels LF 
//- =======================================================================

#pragma once
#ifndef KTRIDIAG_LF_HXX
#define KTRIDIAG_LF_HXX

//---- Header Dependencies -----------------------------------------------

#include "tools/cuvector.hxx"

//---- Function Declaration ----------------------------------------------


int KTridiagLF(float* data, int dir, int N, int M, int batch);

#endif // KTRIDIAG_LF_HXX

