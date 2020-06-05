//- =======================================================================
//+ BPLG Scan KS
//- =======================================================================

//---- Header Dependencies -----------------------------------------------
#include "BPLGScanKS.hxx"
#include "../lib-BPLG/KScanKS.hxx"


//---- Library Header Definitions ----------------------------------------

// direct (dir=1) or inverse (dir=-1)
Transform* BPLGScanKS::calc(int dir) {
	// Load vectors and check pointers
	CUVector<float>* gpuVector = gpuVector_r();
	CUVector<float>::error("Invalid CUDA pointer", !gpuVector);
	float* data = (float*)gpuVector->gpuData();

	

	CUVector<float>* gpuVectorOut = gpuVector_r();
	CUVector<float>::error("Invalid CUDA pointer", !gpuVectorOut);
	float* dataOutput = (float*) gpuVectorOut->gpuData();
	
	
	
	
	int errCode = KScanKS(data,dataOutput, dir, getX(), getY(), getZ());
	CUVector<float>::error("Error launching BPLGScanKS");
	CUVector<float>::sync();
	
	

	
	switch(dir) { 
		case  3:  
			updateCpu_r(); break;
		case -3:  
			updateCpu_r(); break;
	}

	return errCode ? NULL : this;
}
