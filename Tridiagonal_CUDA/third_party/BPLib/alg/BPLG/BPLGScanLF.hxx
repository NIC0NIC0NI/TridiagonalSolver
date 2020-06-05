//- =======================================================================
//+ BPLG Scan KS 
//- =======================================================================

#pragma once
#ifndef BPLG_SCAN_LF_HXX
#define BPLG_SCAN_LF_HXX

//---- Header Dependencies -----------------------------------------------

#include "../Transform.hxx"
#include "tools/cuvector.hxx"

//---- Class Declaration -------------------------------------------------

class BPLGScanLF : public Transform {

public:

	// Creates a transform, batch is the last dimension
	BPLGScanLF(int dimX, int dimY = 1, int dimZ = 1) :
	  Transform(dimX, dimY, dimZ) { init(); }

	// Calculates the transform, direct (dir=1) or inverse (dir=-1)
	Transform* calc(int dir = 1);

	// Initializes data, sequencial (mode=0) or random (mode=1) or ones (mode=4)
	inline Transform* init(int mode = -1) {
		init_r(mode ? 4 : 4);
		updateGpu_r();
		return this;
	}

	// Returns algorithm's name
	inline const char* toString() { return "BPLGScanLF"; }

	// Prints data
	Transform* print(const char* name = 0) {
		updateCpu_r(); print_r(name ? name : toString());
		return this; }






	// Compares relative error with respect to other algorithms
	inline double compare(Transform* ref, const char* label = 0) {
		updateCpu_r(); //? if(ref) ref->updateCpu_c();
		return compare_r(ref, label); }


		// Returns mdata/sec instead of gflops in this case
	inline double gflops(double time, long long iters = 1) const {
		double nProb = (double)iters * (double)getZ();
		double data =  nProb * getX();
		if(time < 1e-3) time = 1e-3; // Preventing division by 0
		return data * 1e-6 / time;
	} 

};

#endif // BPLG_SCAN_LF_HXX
