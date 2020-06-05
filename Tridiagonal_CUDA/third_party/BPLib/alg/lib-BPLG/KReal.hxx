//- =======================================================================
//+ RealFT kernels v1.0
//- =======================================================================

#pragma once
#ifndef KREAL_HXX
#define KREAL_HXX

//---- Header Dependencies -----------------------------------------------

#include "tools/cuvector.hxx"

//---- Function Declaration ----------------------------------------------

int KReal(float2* data, int dir, int N, int M, int batch);

#endif // KREAL_HXX

