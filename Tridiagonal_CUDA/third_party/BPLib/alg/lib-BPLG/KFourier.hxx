//- =======================================================================
//+ Fourier kernels v1.0
//- =======================================================================

#pragma once
#ifndef KFOURIER_HXX
#define KFOURIER_HXX

//---- Header Dependencies -----------------------------------------------

#include "tools/cuvector.hxx"

//---- Function Declaration ----------------------------------------------

int KFourier(float2* data, int dir, int N, int M, int batch);

#endif // KFOURIER_HXX

