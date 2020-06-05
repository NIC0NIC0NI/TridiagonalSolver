//- =======================================================================
//+ Taus generator v1.0
//- =======================================================================

#pragma once
#ifndef TAUSGEN_HXX
#define TAUSGEN_HXX

//---- Function Declaration -----------------------------------------

// Initialize Taus algorithm, if seed is negative full randomization
void tausInit(int seed);

// Generate random 32 bit value using Taus algorithm
unsigned int tausRand();

// Generate signed random value within the given range
unsigned int tausRand(int a, int b);

#endif // TAUSGEN_HXX
