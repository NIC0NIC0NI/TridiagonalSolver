//- =======================================================================
//+ BPLG Tridiagonal PCR Systems v1.0
//- =======================================================================

//---- Header Dependencies -----------------------------------------------
#include "BPLGTridiagPCR.hxx"
#include "../lib-BPLG/KTridiagPCR.hxx"

//---- Library Header Definitions ----------------------------------------

// direct (dir=1) or inverse (dir=-1)
Transform* BPLGTridiagPCR::calc(int dir) {
	// Load vectors and check pointers
	CUVector<float>* gpuVector = gpuVector_r();
	CUVector<float>::error("Invalid CUDA pointer", !gpuVector);
	float* data = (float*)gpuVector->gpuData();

	


	
	int errCode = KTridiagPCR(data, dir, getX(), getY(), getZ());

	CUVector<float>::error("Error launching BPLGTridiagPCR");
	CUVector<float>::sync();


	switch(dir) { 
		case  3:  
			updateCpu_r(); break;
		case -3:  
			updateCpu_r();
	}

	return errCode ? NULL : this;
}
