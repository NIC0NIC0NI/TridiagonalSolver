//- =======================================================================
//+ BPLG Fourier Transform v1.0
//- =======================================================================

//---- Header Dependencies -----------------------------------------------
#include "BPLGFourier.hxx"
#include "../lib-BPLG/KFourier.hxx"

//---- Library Header Definitions ----------------------------------------

// Calcula la transformada, directa (dir=1) o inversa (dir=-1)
Transform* BPLGFourier::calc(int dir) {
	// Cargamos los vectores y comprobamos los punteros
	CUVector<Complex>* gpuVector = gpuVector_c();
	CUVector<Complex>::error("Invalid CUDA pointer", !gpuVector);
	float2* data = (float2*)gpuVector->gpuData();

	// Llamamos al lanzador con los parametros adecuados
	int errCode = KFourier(data, dir, getX(), getY(), getZ());
	CUVector<Complex>::error("Error launching BPLGFFT");
	CUVector<Complex>::sync();

	switch(dir) { // Si es necesario estandarizamos los datos
		case  3:  // Caso directa exacta
		case -3:  // Caso inversa exacta
			updateCpu_c();
	}

	return errCode ? NULL : this;
}

