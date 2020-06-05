
#define NAN __int_as_float(0x7fffffff)

struct Float6;

struct Float4 {

	float x, y, z, w;

	__device__ Float4() { } // Indefined

	__device__ Float4(const float fx, const float fy, const float fz, const float fw) : 
	x(fx), y(fy), z(fz), w(fw) { }

	__device__ Float4(const float* f) :
	x(f[0]), y(f[1]), z(f[2]), w(f[3]) { }

	__device__ Float4(const float4& f) :
	x(f.x), y(f.y), z(f.z), w(f.w) { }
	
	__device__ operator float4() {
		float4 tmp = { x, y, z, w }; return tmp;
	}
	/*
	__device__ inline Float4&
	operator=(const Float4 &f) {
		x = f.x; y = f.y; z = f.z; w = f.w; return *this; }
	*/
	__device__ inline Float4
	operator+(const float &f) const {
		return Float4(x+f, y+f, z+f, w+f); }

	__device__ inline Float4
	operator-(const float &f) const {
		return Float4(x-f, y-f, z-f, w-f); }

	__device__ inline Float4
	operator*(const float &f) const {
		return Float4(x*f, y*f, z*f, w*f); }

	__device__ inline Float4
	operator/(const float &f) const {
		float i = 1.0f / f;
		return Float4(x*i, y*i, z*i, w*i);
	}

	__device__ inline Float4
	operator+(const Float4 &f) const {
		return Float4(x+f.x, y+f.y, z+f.z, w+f.w); }

	__device__ inline Float4
	operator-(const Float4 &f) const {
		return Float4(x-f.x, y-f.y, z-f.z, w-f.w); }

	__device__ inline Float4
	operator*(const Float4 &f) const {
		return Float4(x*f.x, y*f.y, z*f.z, w*f.w); }

	__device__ inline Float4
	operator/(const Float4 &f) const {
		return Float4(x/f.x, y/f.y, z/f.z, w/f.w); }

	__device__ inline Float4
	abs() const {
		return Float4(abs(x), abs(y), abs(z), abs(w)); }

	__device__ inline float
	sum() const {
		return x + y + z + w; }

	__device__ inline float
	dot(const Float4 &f) const {
		return x*f.x + y*f.y + z*f.z + w*f.w;
	}

	__device__ inline float
	max() const { float t1 = x > y ? x : y;
		float t2 = z > w ? z : w; return t1 > t2 ? t1 : t2;
	}

	__device__ inline float
	min() const { float t1 = x < y ? x : y;
		float t2 = z < w ? z : w; return t1 < t2 ? t1 : t2;
	}

	__device__ inline void
	print() const {
		printf("{ % 7.2f, % 7.2f, % 7.2f, % 7.2f }\n", x, y, z, w);
	}

	__device__ inline float
	get(int s) const {
		if(s == 0) return x;
		if(s == 1) return y;
		if(s == 2) return z;
		if(s == 3) return w;
		return NAN;
	}

	__device__ inline bool
	isZero(float eps = 1.0e-5f) const {
		return abs().sum() < eps;
	}

	__device__ inline bool
	isZero(int s, float eps = 1.0e-5f) const {
		return abs(get(s)) < eps;
	}

	__device__ inline bool
	epsComp(const Float4 &f, float eps = 1.0e-5f) const {
		return (*this - f).abs().sum() < eps;
	}

	__device__ Float6 xyz__w() const;

	__device__ Float6 _xy_zw() const;

	__device__ Float6 _x_yzw() const;

	__device__ static inline float abs(const float val) {
		return val < 0.0f ? -val : val; }
};

struct Float4;

struct Float6 {

	float s0, s1, s2, s3, s4, s5;

	__device__ Float6() { } // Indefined

	__device__ Float6(float v0, float v1, float v2, float v3, float v4, float v5) : 
	s0(v0), s1(v1), s2(v2), s3(v3), s4(v4), s5(v5) { }

	__device__ Float6(const float* f) :
	s0(f[0]), s1(f[1]), s2(f[2]), s3(f[3]), s4(f[4]), s5(f[5]) { }

	__device__ inline Float6
	operator*(const float &v) const {
		return Float6(s0*v, s1*v, s2*v, s3*v, s4*v, s5*v); }

	__device__ inline Float6
	operator-(const Float6 &v) const {
		return Float6(s0-v.s0, s1-v.s1, s2-v.s2, s3-v.s3, s4-v.s4, s5-v.s5); }

	__device__ inline Float6
	abs() const {
		return Float6(abs(s0), abs(s1), abs(s2), abs(s3), abs(s4), abs(s5)); }

	__device__ inline float
	sum() const {
		return s0 + s1 + s2 + s3 + s4 + s5; }

	__device__ inline float
	get(int s) const {
		if(s == 0) return s0;
		if(s == 1) return s1;
		if(s == 2) return s2;
		if(s == 3) return s3;
		if(s == 4) return s4;
		if(s == 5) return s5;
		return NAN;
	}

	__device__ inline bool
	isZero(float eps = 1.0e-5f) const {
		return abs().sum() < eps;
	}

	__device__ inline bool
	isZero(int s, float eps = 1.0e-5f) const {
		return abs(get(s)) < eps;
	}

	__device__ inline void
	print() const {
		printf("{ % 7.2f, % 7.2f, % 7.2f, % 7.2f, % 7.2f, %7.2f }\n",
			s0, s1, s2, s3, s4, s5);
	}

	__device__ Float4 s0145() const;

	__device__ Float4 s0345() const;

	__device__ Float4 s0245() const;

	__device__ static inline float abs(const float val) {
		return val < 0.0f ? -val : val; }

};

__device__ Float6 Float4::xyz__w() const { return Float6(x, y, z, 0.0f, 0.0f, w); }

__device__ Float6 Float4::_xy_zw() const { return Float6(0.0f, x, y, 0.0f, z, w); }

__device__ Float6 Float4::_x_yzw() const { return Float6(0.0f, x, 0.0f, y, z, w); }

__device__ Float4 Float6::s0145() const { return Float4(s0, s1, s4, s5); } 

__device__ Float4 Float6::s0345() const { return Float4(s0, s3, s4, s5); }

__device__ Float4 Float6::s0245() const { return Float4(s0, s2, s4, s5); }
