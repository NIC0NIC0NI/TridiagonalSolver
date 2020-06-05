//- =======================================================================
//+ BPLG  Tridiagonal Wang&Mou Systems 
//- =======================================================================

//---- Header Dependencies -----------------------------------------------
#include "BPLGTridiagWM.hxx"
#include "../lib-BPLG/KTridiagWM.hxx"

//---- Library Header Definitions ----------------------------------------

// direct (dir=1) or inverse (dir=-1)
Transform* BPLGTridiagWM::calc(int dir) {
	// Load vectors and check pointers
	CUVector<float>* gpuVector = gpuVector_r();
	CUVector<float>::error("Invalid CUDA pointer", !gpuVector);
	float* data = (float*)gpuVector->gpuData();

	


	
	int errCode = KTridiagWM(data, dir, getX(), getY(), getZ());

	CUVector<float>::error("Error launching BPLGTridiagWM");
	CUVector<float>::sync();


	switch(dir) { 
		case  3:  
			updateCpu_r(); break;
		case -3:  
			updateCpu_r();
	}

	return errCode ? NULL : this;
}
