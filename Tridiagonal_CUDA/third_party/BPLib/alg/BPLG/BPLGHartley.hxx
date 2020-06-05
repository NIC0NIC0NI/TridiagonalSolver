//- =======================================================================
//+ BPLG Hartley Transform v1.0
//- =======================================================================

#pragma once
#ifndef BPLG_HARTLEY_HXX
#define BPLG_HARTLEY_HXX

//---- Header Dependencies -----------------------------------------------

#include "../Transform.hxx"
#include "tools/cuvector.hxx"

//---- Class Declaration -------------------------------------------------

class BPLGHartley : public Transform {

public:

	// Crea una transformada, el batch es la ultima dimension
	BPLGHartley(int dimX, int dimY = 1, int dimZ = 1) :
	  Transform(dimX, dimY, dimZ) { init(); }

	// Calcula la transformada, directa (dir=1) o inversa (dir=-1)
	Transform* calc(int dir = 1);

	// Inicializa los datos, secuencial (mode=0) o aleatorio (mode=1)
	inline Transform* init(int mode = -1) {
		init_r(mode); updateGpu_r(); return this; }

	// Devuelve el nombre del algoritmo
	inline const char* toString() { return "BPLGHartley"; }

	// Imprime los datos por pantalla, etiqueta con nombre opcional
	Transform* print(const char* name = 0) {
		updateCpu_r(); print_r(name ? name : toString());
		return this; }

	// Compara el error relativo con respecto a otro algoritmo
	inline double compare(Transform* ref, const char* label = 0) {
		updateCpu_r(); //? if(ref) ref->updateCpu_c();
		return compare_r(ref, label); }

	// Devuelve el numero de gflops del metodo
	inline double gflops(double time, long long iters = 1) const {
		return gflops_r(time, iters);
	}

};

#endif // BPLG_HARTLEY_HXX
