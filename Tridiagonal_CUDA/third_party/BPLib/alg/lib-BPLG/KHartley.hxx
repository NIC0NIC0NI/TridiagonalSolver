//- =======================================================================
//+ Hartley kernels v1.0
//- =======================================================================

#pragma once
#ifndef KHARTLEY_HXX
#define KHARTLEY_HXX

//---- Header Dependencies -----------------------------------------------

#include "tools/cuvector.hxx"

//---- Function Declaration ----------------------------------------------

int KHartley(float* data, int dir, int N, int M, int batch);

#endif // KHARTLEY_HXX

