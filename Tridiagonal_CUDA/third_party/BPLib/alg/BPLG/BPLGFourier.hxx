//- =======================================================================
//+ BPLG Fourier Transform v1.0
//- =======================================================================

#pragma once
#ifndef BPLG_FOURIER_HXX
#define BPLG_FOURIER_HXX

//---- Header Dependencies -----------------------------------------------

#include "../Transform.hxx"
#include "tools/cuvector.hxx"

//---- Class Declaration -------------------------------------------------

class BPLGFourier : public Transform {

public:

	// Crea una transformada, el batch es la ultima dimension
	BPLGFourier(int dimX, int dimY = 1, int dimZ = 1) :
	  Transform(dimX, dimY, dimZ) { init(); }

	// Calcula la transformada, directa (dir=1) o inversa (dir=-1)
	Transform* calc(int dir = 1);

	// Inicializa los datos, secuencial (mode=0) o aleatorio (mode=1)
	inline Transform* init(int mode = -1) {
		init_c(mode); updateGpu_c(); return this; }

	// Devuelve el nombre del algoritmo
	inline const char* toString() { return "BPLGFourier"; }

	// Imprime los datos por pantalla, etiqueta con nombre opcional
	inline Transform* print(const char* name = 0) {
		updateCpu_c(); print_c(name ? name : toString());
		return this; }

	// Compara el error relativo con respecto a otro algoritmo
	inline double compare(Transform* ref, const char* label = 0) {
		updateCpu_c(); //? if(ref) ref->updateCpu_c();
		return compare_c(ref, label); }

	// Devuelve el numero de gflops del metodo
	inline double gflops(double time, long long iters = 1) const {
		return gflops_c(time, iters); }

};

#endif // BPLG_FOURIER_HXX
