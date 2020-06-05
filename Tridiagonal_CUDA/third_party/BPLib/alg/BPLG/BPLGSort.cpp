//- =======================================================================
//+ BPLG Sort
//- =======================================================================

//---- Header Dependencies -----------------------------------------------
#include "BPLGSort.hxx"
#include "../lib-BPLG/KSort.hxx"

//---- Library Header Definitions ----------------------------------------

// direct (dir=1) or inverse (dir=-1)
Transform* BPLGSort::calc(int dir) {
	// Load vectors and check pointers
	CUVector<int>* gpuVector = gpuVector_();
	CUVector<int>::error("Invalid CUDA pointer", !gpuVector);
	int* data = (int*)gpuVector->gpuData();

	

	
	
	int errCode = KSort(data, dir, getX(), getY(), getZ());
	CUVector<int>::error("Error launching BPLGSort");
	CUVector<int>::sync();
	
	
	

	
	switch(dir) { 
		case  3:  
			updateCpu_(); break;
		case -3:  
			updateCpu_(); break;
	}

	return errCode ? NULL : this;
}
