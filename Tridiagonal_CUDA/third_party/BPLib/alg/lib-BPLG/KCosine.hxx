//- =======================================================================
//+ Cosine kernels
//- =======================================================================

#pragma once
#ifndef KCOSINE_HXX
#define KCOSINE_HXX

//---- Header Dependencies -----------------------------------------------

#include "tools/cuvector.hxx"

//---- Function Declaration ----------------------------------------------

int KCosine(float* data, int dir, int N, int M, int batch);

#endif // KCOSINE_HXX

