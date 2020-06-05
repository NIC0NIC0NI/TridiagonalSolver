//- =======================================================================
//+ BPLG Hartley Transform v1.0
//- =======================================================================

//---- Header Dependencies -----------------------------------------------
#include "BPLGHartley.hxx"
#include "../lib-BPLG/KHartley.hxx"

//---- Library Header Definitions ----------------------------------------

// Calcula la transformada, directa (dir=1) o inversa (dir=-1)
Transform* BPLGHartley::calc(int dir) {
	// Cargamos los vectores y comprobamos los punteros
	CUVector<float>* gpuVector = gpuVector_r();
	CUVector<float>::error("Invalid CUDA pointer", !gpuVector);
	float* data = (float*)gpuVector->gpuData();

	// Llamamos al lanzador con los parametros adecuados
	int errCode = KHartley(data, dir, getX(), getY(), getZ());
	CUVector<float>::error("Error launching BPLGFFT");
	CUVector<float>::sync();

	switch(dir) { // Si es necesario estandarizamos los datos
		case  3:  // Caso directa exacta
		case -3:  // Caso inversa exacta
			updateCpu_r();
	}

	return errCode ? NULL : this;
}
