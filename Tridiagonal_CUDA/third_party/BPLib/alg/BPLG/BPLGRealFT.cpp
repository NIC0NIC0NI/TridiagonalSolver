//- =======================================================================
//+ BPLG RealFT Transform v1.0
//- =======================================================================

//---- Header Dependencies -----------------------------------------------
#include "BPLGRealFT.hxx"
#include "../lib-BPLG/KReal.hxx"

//---- Library Header Definitions ----------------------------------------

void BPLGRealFT::updateR(int dir) {
	int halfN = swap->getX();
	int batch = swap->getY() * swap->getZ();
	if(dir < 0) swap->cpuRead();
	for(int j = 0; j < batch; j++) {
		// Punteros a datos
		Complex* buffer = swap->cpuData() + j * halfN;
		float* rdata = getData_r() + 2 * j * halfN;

		if(dir > 0) { // Version forward
			for(int i = 0; i < halfN; i++)
				buffer[i] = Complex(rdata[2*i], rdata[2*i+1]);
		}
		if(dir < 0) { // Version inversa
			for(int i = 0; i < halfN; i++) {
				rdata[2*i  ] = buffer[i].real();
				rdata[2*i+1] = buffer[i].imag();
			}
		}
	}
	if(dir > 0) swap->gpuRead();
}

void BPLGRealFT::updateC(int dir) {
	int halfN = swap->getX();
	int batch = swap->getY() * swap->getZ();
	if(dir < 0) swap->cpuRead();
	for(int j = 0; j < batch; j++) {
		// Punteros a datos
		Complex* buffer = swap->cpuData() + j * halfN;
		Complex* cdata = getData_c() + 2 * j * halfN;

		if(dir > 0) { // Version forward
			for(int i = 1; i < halfN; i++)
				buffer[i] = cdata[i];
			buffer[0] = Complex(cdata[0].real(), cdata[halfN].real());
		}
		if(dir < 0) { // Version inversa
			for(int i = 1; i < halfN; i++) {
				cdata[i      ] = buffer[i];
				cdata[i+halfN] = conj(buffer[halfN-i]);
			}
			cdata[0] = Complex(buffer[0].real(), 0.0f);
			cdata[halfN] = Complex(buffer[0].imag(), 0.0f);
		}
	}
	if(dir > 0) swap->gpuRead();
}

// Inicializa los datos, secuencial (mode=0) o aleatorio (mode=1)
Transform* BPLGRealFT::init(int mode) {
	init_r(mode); updateGpu_r();
	init_c(-1); updateGpu_c();
	if(!swap) swap = new CUVector<Complex>(getX() >> 1, getY(), getZ());
	updateR(1);
	last_dir = -1;
	return this;
}

// Imprime los datos por pantalla, etiqueta con nombre opcional
Transform* BPLGRealFT::print(const char* name) {
	if(!name) name = toString();
	if(last_dir < 0) {
		updateR(-1);
		print_r(name);
	} else {
		updateC(-1);
		print_c(name);
	}
	return this;
}

// Compara el error relativo con respecto a otro algoritmo
double BPLGRealFT::compare(Transform* ref, const char* label) {
	if(last_dir < 0) {
		updateR(-1);
		return compare_r(ref,label);
	} else {
		updateC(-1);
		return compare_c(ref, label);
	}
}

// Calcula la transformada, directa (dir=1) o inversa (dir=-1)
Transform* BPLGRealFT::calc(int dir) {
	// Comprobamos que sea una transformada 1D
	CUVector<Complex>::error("Unsupported dimension for rFFT", getY() > 1);

	// Llamamos al lanzador con los parametros adecuados
	int errCode = KReal((float2*)swap->gpuData(),
		dir, swap->getX(), swap->getY(), swap->getZ());
	CUVector<Complex>::error("Error launching LegaFFT");
	CUVector<Complex>::sync();

	switch(dir) { // Si es necesario estandarizamos los datos
		case  3: updateC(-1); break;  // Caso directa exacta
		case -3: updateR(-1); break;  // Caso inversa exacta
	}

	last_dir = dir;
	return errCode ? NULL : this;
}

