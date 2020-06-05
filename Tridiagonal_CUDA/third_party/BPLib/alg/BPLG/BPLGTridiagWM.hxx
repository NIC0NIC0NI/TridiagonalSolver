//- =======================================================================
//+ BPLG Tridiagonal WM Systems
//- =======================================================================

#pragma once
#ifndef BPLG_TRIDIAG_WM_HXX
#define BPLG_TRIDIAG_WM_HXX

//---- Header Dependencies -----------------------------------------------

#include "../Transform.hxx"
#include "tools/cuvector.hxx"

//---- Class Declaration -------------------------------------------------

class BPLGTridiagWM : public Transform {

public:

	// Creates a transform, batch is the last dimension
	BPLGTridiagWM(int dimX, int dimY = 1, int dimZ = 1) :
	  Transform(dimX, dimY, dimZ) { init(); }

	// Calculates the transform, direct (dir=1) or inverse (dir=-1)
	Transform* calc(int dir = 1);

	// Initializes data with numerical stabillity, diagonally dominant
	inline Transform* init(int mode = -1) {
		
		
		init_r(-1);
		
		CUVector<float>* data = gpuVector_r();

		for(int j=0;j<getZ();j++){
			for(int i=0;i<getX();i++){
				(*data)(i,0,j)= -1.0f;
				(*data)(i,1,j)= 2.0f;
				(*data)(i,2,j)= -1.0f;
				(*data)(i,3,j)= 0.0f;
			}
			(*data)(0,0,j)=0.0;
			(*data)(getX()-1,2,j)=0.0;
			(*data)(0,3,j)=1.0f;
			(*data)(getX()-1,3,j)=1.0f;
		}
		updateGpu_r();
		updateCpu_r();
		
		/*
		init_r(mode ? 3 : 3);
		updateGpu_r();*/
		return this;
	}

	// Returns algorithm's name
	inline const char* toString() { return "BPLGTridiagWM"; }

	// Prints data
	Transform* print(const char* name = 0) {
		updateCpu_r(); print_r(name ? name : toString());
		return this; }

	// Compares relative error with respect to other algorithms
	inline double compare(Transform* ref, const char* label = 0) {
		updateCpu_r(); //? if(ref) ref->updateCpu_c();
		return compare_r(ref, label); }

	// Returns mrows/sec instead of gflops in this case
	inline double gflops(double time, long long iters = 1) const {
		double nProb = (double)iters * (double)getZ();
		double data =  nProb * getX();
		if(time < 1e-3) time = 1e-3; // Preventing division by 0
		return data * 1e-6 / time;
	}  


};

#endif // BPLG_TRIDIAG_WM_HXX

