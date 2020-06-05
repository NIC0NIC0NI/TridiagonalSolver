
//- =======================================================================
//+ Reduce
//- =======================================================================

#pragma once
#ifndef _OP_REDUCE
#define _OP_REDUCE



//---- Equation class ----------------------------------------------------

//- Equation class, encapsulates type independent short vectors like Eqn<DTYPE>
template<class DTYPE> struct __builtin_align__(16) Eqn {

	// Default constructor: the object is created with undefined values
	__device__ Eqn() { }

	// The equation object is created with the specified values
	__device__ Eqn(const DTYPE &v1, const DTYPE &v2 = 0,
		const DTYPE &v3 = 0, const DTYPE &v4 = 0) :
		x(v1), y(v2), z(v3), w(v4) { }

	__device__ Eqn(const float4 &v4) :
		x(v4.x), y(v4.y), z(v4.z), w(v4.w) { }

	__device__ Eqn(const double4 &v4) :
		x(v4.x), y(v4.y), z(v4.z), w(v4.w) { }

	__device__ operator float4() const {
		return make_float4(x, y, z, w); }

	__device__ operator double4() const {
		return make_double4(x, y, z, w); }

	// Components of the equations, directly accesible
	DTYPE x, y, z, w;
};

//---- Reduce operator ---------------------------------------------------

/// __device__ const float epsilon = 1e-5f;

//- Reduce ecuaciones con un formato conocido y predeterminado
//- Cada 'Eqn<DTYPE>' representa una ecuacion. Al no disponer de informacion
//- sobre la posicion se usa el parametro 'pos' (ver comentarios)
template<class DTYPE> __device__ Eqn<DTYPE>
reduce(const Eqn<DTYPE> &eq1, const Eqn<DTYPE> &eq2, int pos) {
	switch(pos) {
		case 0: { //! [ a b C _ | s ] % [ _ b C d | s ] = [ a b _ d | s ]
			DTYPE div = eq2.y; // eq1.z
			// if(div > -epsilon && div < epsilon) return eq1;
			DTYPE fac = eq1.z / div;
			return Eqn<DTYPE>(eq1.x, eq1.y -fac * eq2.x, -fac * eq2.z, eq1.w -fac * eq2.w);
		}

		case 1: { //! [ a b C _ | s ] % [ a _ C d | s ] = [ a b _ d | s ]
			DTYPE div = eq2.y; // eq1.z
			// if(div > -epsilon && div < epsilon) return eq1;
			DTYPE fac = eq1.z / div;
			return Eqn<DTYPE>(eq1.x -fac * eq2.x, eq1.y, -fac * eq2.z, eq1.w -fac * eq2.w);
		}

		case 2: { //! [ a B c _ | s ] % [ _ B c d | s ] = [ a _ c d | s ]
			DTYPE div = eq2.x; // eq1.y
			// if(div > -epsilon && div < epsilon) return eq1;
			DTYPE fac = eq1.y / div;
			return Eqn<DTYPE>(eq1.x, eq1.z -fac * eq2.y, -fac * eq2.z, eq1.w -fac * eq2.w);
		}

		case 3: { //! [ a B _ d | s ] % [ _ B c d | s ] = [ a _ c d | s ]
			DTYPE div = eq2.x; // eq1.y
			// if(div > -epsilon && div < epsilon) return eq1;
			DTYPE fac = eq1.y / div;
			return Eqn<DTYPE>(eq1.x, -fac * eq2.y, eq1.z -fac * eq2.z, eq1.w -fac * eq2.w);
		}

		case 4: { //! [ a B c _ | s ] % [ a B _ d | s ] = [ a _ c d | s ]
			DTYPE div = eq2.y; // eq1.y
			// if(div > -epsilon && div < epsilon) break;
			DTYPE fac = eq1.y / div;
			return Eqn<DTYPE>(eq1.x -fac * eq2.x, eq1.z, -fac * eq2.z, eq1.w -fac * eq2.w);
		}

		case 5: { //! [ a _ C d | s ] % [ _ b C d | s ] = [ a b _ d | s ]
			DTYPE div = eq2.y; // eq1.y
			// if(div > -epsilon && div < epsilon) break;
			DTYPE fac = eq1.y / div;
			return Eqn<DTYPE>(eq1.x, -fac * eq2.x, eq1.z -fac * eq2.z, eq1.w -fac * eq2.w);
		}

		case 6: { //! [ a B _ d | s ] % [ a B c _ | s ] = [ a _ c d | s ]
			DTYPE div = eq2.y; // eq1.y
			// if(div > -epsilon && div < epsilon) break;
			DTYPE fac = eq1.y / div;
			return Eqn<DTYPE>(eq1.x -fac * eq2.x, -fac * eq2.z, eq1.z, eq1.w -fac * eq2.w);
		}

		case 7: { //! [ a _ C d | s ] % [ a b C _ | s ] = [ a b _ d | s ]
			DTYPE div = eq2.z; // eq1.y
			// if(div > -epsilon && div < epsilon) break;
			DTYPE fac = eq1.y / div;
			return Eqn<DTYPE>(eq1.x -fac * eq2.x, -fac * eq2.y, eq1.z, eq1.w -fac * eq2.w);
		}

		case 8: { //! [ _ b C d | s ] % [ a b C _ | s ] = [ a b _ d | s ]
			DTYPE div = eq2.z; // eq1.y
			// if(div > -epsilon && div < epsilon) break;
			DTYPE fac = eq1.y / div;
			return Eqn<DTYPE>(-fac * eq2.x, eq1.x -fac * eq2.y, eq1.z, eq1.w -fac * eq2.w);
		}

		case 9: { //! [ _ B c d | s ] % [ a B _ d | s ] = [ a _ c d | s ]
			DTYPE div = eq2.y; // eq1.x
			// if(div > -epsilon && div < epsilon) break;
			DTYPE fac = eq1.x / div;
			return Eqn<DTYPE>(-fac * eq2.x, eq1.y, eq1.z -fac * eq2.z, eq1.w -fac * eq2.w);
		}

		case 10: { //! [ _ B c d | s ] % [ a B c _ | s ] = [ a b _ d | s ]
			DTYPE div = eq2.y; // eq1.x
			// if(div > -epsilon && div < epsilon) break;
			DTYPE fac = eq1.x / div;
			return Eqn<DTYPE>(-fac * eq2.x, eq1.y -fac * eq2.z, eq1.z, eq1.w -fac * eq2.w);
		}
		
		case 11: { //! [ a b _ D | s ] % [ _ b c D | s ] = [ a b c _ | s ]

			DTYPE div = eq1.y;

			DTYPE fac = -eq2.x/ div;
			return Eqn<DTYPE>(eq1.x * fac, eq2.y, eq2.z + eq1.z * fac, eq2.w + fac * eq1.w);
		}
		

	}
	return eq1;
}

#endif // _OP_REDUCE
