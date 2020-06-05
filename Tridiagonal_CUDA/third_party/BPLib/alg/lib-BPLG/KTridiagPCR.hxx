//- =======================================================================
//+ BPLG Tridiagonal PCR kernels 
//- =======================================================================

#pragma once
#ifndef KTRIDIAG_PCR_HXX
#define KTRIDIAG_PCR_HXX

//---- Header Dependencies -----------------------------------------------

#include "tools/cuvector.hxx"

//---- Function Declaration ----------------------------------------------

//- Kernel para sistemas de ecuaciones tridiagonales reales
int KTridiagPCR(float* data, int dir, int N, int M, int batch);


#endif // KTRIDIAG_PCR_HXX

