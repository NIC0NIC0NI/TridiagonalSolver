//- =======================================================================
//+ BPLG Tridiagonal Systems LF 
//- =======================================================================

//---- Header Dependencies -----------------------------------------------
#include "BPLGTridiagLF.hxx"
#include "../lib-BPLG/KTridiagLF.hxx"

//---- Library Header Definitions ----------------------------------------

// direct (dir=1) or inverse (dir=-1)
Transform* BPLGTridiagLF::calc(int dir) {
	// Load vectors and check pointers
	CUVector<float>* gpuVector = gpuVector_r();
	CUVector<float>::error("Invalid CUDA pointer", !gpuVector);
	float* data = (float*)gpuVector->gpuData();

	int errCode = KTridiagLF(data, dir, getX(), getY(), getZ());
	CUVector<float>::error("Error launching BPLGTridiagLF");
	CUVector<float>::sync();
	

	
	switch(dir) { 
		case  3:  
			updateCpu_r(); break;
		case -3:  
			updateCpu_r();
	}

	return errCode ? NULL : this;
}
