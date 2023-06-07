#ifndef dDimensionalVectorTypes_h
#define dDimensionalVectorTypes_h
/*! \file dDimensionalVectorTypes.h
defines dVec class (d-dimensional array of scalars)
defines iVec class (d-dimensional array of ints)
*/

//#ifndef SCALARFLOAT
//double variables types
//#define scalar double
//#else
#define scalar float
//#endif

//#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
//#define MY_ALIGN(n) __align__(n)
//#else
//#define HOSTDEVICE inline __attribute__((always_inline))
#define MY_ALIGN(n) __attribute__((aligned(n)))
//#endif

//!dVec is an array whose length matches the dimension of the system
class MY_ALIGN(8) dVec
    {
    public:
        HOSTDEVICE dVec(){};
        HOSTDEVICE dVec(const scalar value)
            {
            for (int dd = 0; dd < DIMENSION; ++dd)
                x[dd] = value;
            };
        HOSTDEVICE dVec(const dVec &other)
            {
            for (int dd = 0; dd < DIMENSION; ++dd)
                x[dd] = other.x[dd];
            };

        scalar x[DIMENSION];

        HOSTDEVICE scalar& operator[](int i){return x[i];};

        //mutating operators
        HOSTDEVICE dVec& operator=(const dVec &other)
            {
            for (int dd = 0; dd < DIMENSION; ++dd)
                this->x[dd] = other.x[dd];
            return *this;
            }
        HOSTDEVICE dVec& operator-=(const dVec &other)
            {
            for (int dd = 0; dd < DIMENSION; ++dd)
                this->x[dd] -= other.x[dd];
            return *this;
            }
        HOSTDEVICE dVec& operator+=(const dVec &other)
            {
            for (int dd = 0; dd < DIMENSION; ++dd)
                this->x[dd] += other.x[dd];
            return *this;
            }
    };


//!Less than operator for dVecs just sorts by the x-coordinate
HOSTDEVICE bool operator<(const dVec &a, const dVec &b)
    {
    return a.x[0]<b.x[0];
    }

//!Equality operator tests for.... equality of all elements
HOSTDEVICE bool operator==(const dVec &a, const dVec &b)
    {
    for (int dd = 0; dd <DIMENSION; ++dd)
        if(a.x[dd]!= b.x[dd]) return false;
    return true;
    }

//!return a dVec with all elements equal to one number
HOSTDEVICE dVec make_dVec(scalar value)
    {
    dVec ans;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans.x[dd] = value;
    return ans;
    }

//!component-wise addition of two dVecs
HOSTDEVICE dVec operator+(const dVec &a, const dVec &b)
    {
    dVec ans;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans.x[dd] = a.x[dd]+b.x[dd];
    return ans;
    }

//!component-wise subtraction of two dVecs
HOSTDEVICE dVec operator-(const dVec &a, const dVec &b)
    {
    dVec ans;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans.x[dd] = a.x[dd]-b.x[dd];
    return ans;
    }

//!component-wise multiplication of two dVecs
HOSTDEVICE dVec operator*(const dVec &a, const dVec &b)
    {
    dVec ans;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans.x[dd] = a.x[dd]*b.x[dd];
    return ans;
    }

//!multiplication of dVec by scalar
HOSTDEVICE dVec operator*(const scalar &a, const dVec &b)
    {
    dVec ans;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans.x[dd] = a*b.x[dd];
    return ans;
    }

//!multiplication of dVec by scalar
HOSTDEVICE dVec operator*(const dVec &b, const scalar &a)
    {
    dVec ans;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans.x[dd] = a*b.x[dd];
    return ans;
    }

//!print a dVec to screen
inline __attribute__((always_inline)) void printdVecListable(dVec a)
    {
    std::cout <<"{";
    for (int dd = 0; dd < DIMENSION; ++dd)
        if(dd != DIMENSION-1)
            std::cout << a.x[dd] <<", ";
        else
            std::cout << a.x[dd];

    std::cout << "},";
    };
//!print a dVec to screen
inline __attribute__((always_inline)) void printdVec(dVec a)
    {
    std::cout <<"{";
    for (int dd = 0; dd < DIMENSION; ++dd)
        if(dd != DIMENSION-1)
            std::cout << a.x[dd] <<", ";
        else
            std::cout << a.x[dd];

    std::cout << "}" << std::endl;
    };


#undef HOSTDEVICE
#endif
