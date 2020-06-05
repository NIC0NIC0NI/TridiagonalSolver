//- =======================================================================
//+ Taus generator v1.0
//- =======================================================================

//---- Header File -------------------------------------------------------
#include "tausgen.hxx"

//---- Include Section ---------------------------------------------------
#include <ctime>
#include <cstdlib>

//---- Library Code ------------------------------------------------------

// Status storage variables
static unsigned int _s1 = 0x6238553B;
static unsigned int _s2 = 0xA2B92C07;
static unsigned int _s3 = 0x7D36B8C4;

// Generator initialization
void tausInit(int seed) {
	if(seed < 0) seed = (unsigned int)time(0);
	srand(seed);
	_s1 = rand();
	_s2 = rand();
	_s3 = rand();
}

// Generate random 32 bit value
unsigned int tausRand() {
	_s1 = ((_s1 & 0xfffffffe) << 12) ^ (((_s1 << 13) ^ _s1) >> 19);
	_s2 = ((_s2 & 0xfffffff8) <<  4) ^ (((_s2 <<  2) ^ _s2) >> 25);
	_s3 = ((_s3 & 0xfffffff0) << 17) ^ (((_s3 <<  3) ^ _s3) >> 11);
	return _s1 ^ _s2 ^ _s3;
}

// Generate signed random value within the given range
unsigned int tausRand(int a, int b) {
	int hi = a > b ? a : b;
	int lo = a < b ? a : b;
	int range = hi - lo;
	return (tausRand() % range) + lo;
}