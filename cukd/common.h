// ======================================================================== //
// Copyright 2018-2022 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

/* copied from OWL project, and put into new namespace to avoid naming conflicts.*/

#pragma once

#ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES
#endif
#include <math.h> // using cmath causes issues under Windows
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <assert.h>
#include <string>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <sstream>
#ifdef __GNUC__
#include <execinfo.h>
#include <sys/time.h>
#endif

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

#if !defined(WIN32)
#include <signal.h>
#endif

#if defined(_MSC_VER)
#  define CUKD_DLL_EXPORT __declspec(dllexport)
#  define CUKD_DLL_IMPORT __declspec(dllimport)
#elif defined(__clang__) || defined(__GNUC__)
#  define CUKD_DLL_EXPORT __attribute__((visibility("default")))
#  define CUKD_DLL_IMPORT __attribute__((visibility("default")))
#else
#  define CUKD_DLL_EXPORT
#  define CUKD_DLL_IMPORT
#endif

// #if 1
# define CUKD_INTERFACE /* nothing - currently not building any special 'owl.dll' */
// #else
// //#if defined(CUKD_DLL_INTERFACE)
// #  ifdef owl_EXPORTS
// #    define CUKD_INTERFACE CUKD_DLL_EXPORT
// #  else
// #    define CUKD_INTERFACE CUKD_DLL_IMPORT
// #  endif
// //#else
// //#  define CUKD_INTERFACE /*static lib*/
// //#endif
// #endif

//#ifdef __WIN32__
//#define  __PRETTY_FUNCTION__ __FUNCTION__
//#endif
#if defined(_MSC_VER)
//&& !defined(__PRETTY_FUNCTION__)
#  define __PRETTY_FUNCTION__ __FUNCTION__
#endif


#ifndef PRINT
# define PRINT(var) std::cout << #var << "=" << var << std::endl;
#ifdef __WIN32__
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __FUNCTION__ << std::endl;
#else
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __PRETTY_FUNCTION__ << std::endl;
#endif
#endif

#if defined(__CUDA_ARCH__)
# define __owl_device   __device__
# define __owl_host     __host__
#else
# define __owl_device   /* ignore */
# define __owl_host     /* ignore */
#endif

# define __both__   __owl_host __owl_device


#ifdef __GNUC__
#define MAYBE_UNUSED __attribute__((unused))
#else
#define MAYBE_UNUSED
#endif

#define CUKD_NOTIMPLEMENTED throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+" not implemented")

#ifdef WIN32
# define CUKD_TERMINAL_RED ""
# define CUKD_TERMINAL_GREEN ""
# define CUKD_TERMINAL_LIGHT_GREEN ""
# define CUKD_TERMINAL_YELLOW ""
# define CUKD_TERMINAL_BLUE ""
# define CUKD_TERMINAL_LIGHT_BLUE ""
# define CUKD_TERMINAL_RESET ""
# define CUKD_TERMINAL_DEFAULT CUKD_TERMINAL_RESET
# define CUKD_TERMINAL_BOLD ""

# define CUKD_TERMINAL_MAGENTA ""
# define CUKD_TERMINAL_LIGHT_MAGENTA ""
# define CUKD_TERMINAL_CYAN ""
# define CUKD_TERMINAL_LIGHT_RED ""
#else
# define CUKD_TERMINAL_RED "\033[0;31m"
# define CUKD_TERMINAL_GREEN "\033[0;32m"
# define CUKD_TERMINAL_LIGHT_GREEN "\033[1;32m"
# define CUKD_TERMINAL_YELLOW "\033[1;33m"
# define CUKD_TERMINAL_BLUE "\033[0;34m"
# define CUKD_TERMINAL_LIGHT_BLUE "\033[1;34m"
# define CUKD_TERMINAL_RESET "\033[0m"
# define CUKD_TERMINAL_DEFAULT CUKD_TERMINAL_RESET
# define CUKD_TERMINAL_BOLD "\033[1;1m"

# define CUKD_TERMINAL_MAGENTA "\e[35m"
# define CUKD_TERMINAL_LIGHT_MAGENTA "\e[95m"
# define CUKD_TERMINAL_CYAN "\e[36m"
# define CUKD_TERMINAL_LIGHT_RED "\033[1;31m"
#endif

#ifdef _MSC_VER
# define CUKD_ALIGN(alignment) __declspec(align(alignment))
#else
# define CUKD_ALIGN(alignment) __attribute__((aligned(alignment)))
#endif



namespace cukd {
  namespace common {

#ifdef __CUDA_ARCH__
    using ::min;
    using ::max;
    using std::abs;
#else
    using std::min;
    using std::max;
    using std::abs;
    inline __both__ float saturate(const float &f) { return min(1.f,max(0.f,f)); }
#endif

    inline __both__ float rcp(float f)      { return 1.f/f; }
    inline __both__ double rcp(double d)    { return 1./d; }

    inline __both__ int32_t divRoundUp(int32_t a, int32_t b) { return (a+b-1)/b; }
    inline __both__ uint32_t divRoundUp(uint32_t a, uint32_t b) { return (a+b-1)/b; }
    inline __both__ int64_t divRoundUp(int64_t a, int64_t b) { return (a+b-1)/b; }
    inline __both__ uint64_t divRoundUp(uint64_t a, uint64_t b) { return (a+b-1)/b; }

    using ::sin; // this is the double version
    using ::cos; // this is the double version

    /*! namespace that offers polymorphic overloads of functions like
      sqrt, rsqrt, sin, cos, etc (that vary based on float vs
      double), and that is NOT in a default namespace where ti
      would/could clash with cuda or system-defines of the same name
      - TODO: make sure that cos, sin, abs, etc are also properly
      handled here. */
    namespace polymorphic {
#ifdef __CUDA_ARCH__
      inline __both__ float sqrt(const float f)     { return ::sqrtf(f); }
      inline __both__ double sqrt(const double d)   { return ::sqrt(d); }
#else
      inline __both__ float sqrt(const float f)     { return ::sqrtf(f); }
      inline __both__ double sqrt(const double d)   { return ::sqrt(d); }
#endif

      inline __both__ float rsqrt(const float f)    { return 1.f/cukd::common::polymorphic::sqrt(f); }
      inline __both__ double rsqrt(const double d)  { return 1./cukd::common::polymorphic::sqrt(d); }
    }


#ifdef __WIN32__
#  define osp_snprintf sprintf_s
#else
#  define osp_snprintf snprintf
#endif

    /*! added pretty-print function for large numbers, printing 10000000 as "10M" instead */
    inline std::string prettyDouble(const double val) {
      const double absVal = abs(val);
      char result[1000];

      if      (absVal >= 1e+18f) osp_snprintf(result,1000,"%.1f%c",float(val/1e18f),'E');
      else if (absVal >= 1e+15f) osp_snprintf(result,1000,"%.1f%c",float(val/1e15f),'P');
      else if (absVal >= 1e+12f) osp_snprintf(result,1000,"%.1f%c",float(val/1e12f),'T');
      else if (absVal >= 1e+09f) osp_snprintf(result,1000,"%.1f%c",float(val/1e09f),'G');
      else if (absVal >= 1e+06f) osp_snprintf(result,1000,"%.1f%c",float(val/1e06f),'M');
      else if (absVal >= 1e+03f) osp_snprintf(result,1000,"%.1f%c",float(val/1e03f),'k');
      else if (absVal <= 1e-12f) osp_snprintf(result,1000,"%.1f%c",float(val*1e15f),'f');
      else if (absVal <= 1e-09f) osp_snprintf(result,1000,"%.1f%c",float(val*1e12f),'p');
      else if (absVal <= 1e-06f) osp_snprintf(result,1000,"%.1f%c",float(val*1e09f),'n');
      else if (absVal <= 1e-03f) osp_snprintf(result,1000,"%.1f%c",float(val*1e06f),'u');
      else if (absVal <= 1e-00f) osp_snprintf(result,1000,"%.1f%c",float(val*1e03f),'m');
      else osp_snprintf(result,1000,"%f",(float)val);

      return result;
    }


    /*! return a nicely formatted number as in "3.4M" instead of
      "3400000", etc, using mulitples of thousands (K), millions
      (M), etc. Ie, the value 64000 would be returned as 64K, and
      65536 would be 65.5K */
    inline std::string prettyNumber(const size_t s)
    {
      char buf[1000];
      if (s >= (1000LL*1000LL*1000LL*1000LL)) {
        osp_snprintf(buf, 1000,"%.2fT",s/(1000.f*1000.f*1000.f*1000.f));
      } else if (s >= (1000LL*1000LL*1000LL)) {
        osp_snprintf(buf, 1000, "%.2fG",s/(1000.f*1000.f*1000.f));
      } else if (s >= (1000LL*1000LL)) {
        osp_snprintf(buf, 1000, "%.2fM",s/(1000.f*1000.f));
      } else if (s >= (1000LL)) {
        osp_snprintf(buf, 1000, "%.2fK",s/(1000.f));
      } else {
        osp_snprintf(buf,1000,"%zi",s);
      }
      return buf;
    }

    /*! return a nicely formatted number as in "3.4M" instead of
      "3400000", etc, using mulitples of 1024 as in kilobytes,
      etc. Ie, the value 65534 would be 64K, 64000 would be 63.8K */
    inline std::string prettyBytes(const size_t s)
    {
      char buf[1000];
      if (s >= (1024LL*1024LL*1024LL*1024LL)) {
        osp_snprintf(buf, 1000,"%.2fT",s/(1024.f*1024.f*1024.f*1024.f));
      } else if (s >= (1024LL*1024LL*1024LL)) {
        osp_snprintf(buf, 1000, "%.2fG",s/(1024.f*1024.f*1024.f));
      } else if (s >= (1024LL*1024LL)) {
        osp_snprintf(buf, 1000, "%.2fM",s/(1024.f*1024.f));
      } else if (s >= (1024LL)) {
        osp_snprintf(buf, 1000, "%.2fK",s/(1024.f));
      } else {
        osp_snprintf(buf,1000,"%zi",s);
      }
      return buf;
    }

    inline double getCurrentTime()
    {
#ifdef _WIN32
      SYSTEMTIME tp; GetSystemTime(&tp);
      /*
        Please note: we are not handling the "leap year" issue.
      */
      size_t numSecsSince2020
        = tp.wSecond
        + (60ull) * tp.wMinute
        + (60ull * 60ull) * tp.wHour
        + (60ull * 60ul * 24ull) * tp.wDay
        + (60ull * 60ul * 24ull * 365ull) * (tp.wYear - 2020);
      return double(numSecsSince2020 + tp.wMilliseconds * 1e-3);
#else
      struct timeval tp; gettimeofday(&tp,nullptr);
      return double(tp.tv_sec) + double(tp.tv_usec)/1E6;
#endif
    }

    inline bool hasSuffix(const std::string &s, const std::string &suffix)
    {
      return s.substr(s.size()-suffix.size()) == suffix;
    }

    template<typename T> struct point_traits;

    template<> struct point_traits<float3> { enum { numDims = 3 }; };
    template<> struct point_traits<float4> { enum { numDims = 4 }; };

    /*! Trivial implementation of the point interface for those kinds of
      point types where the first K elements are the K-dimensional
      coordinates that we buid the K-d tree over; the point_t struct
      may contain additional data at the end, too (ie, you can build,
      for exapmle, a 2-d tree over a float4 point type - in this case
      the x and y coordinates are the point coordinates, and z and w
      are any other payload that does not get considered during the
      (2-d) construction) */
    template<typename _point_t, typename scalar_t>
    struct TrivialPointInterface
    {
      typedef _point_t point_t;
      inline static __host__ __device__
      scalar_t get(const point_t &p, int dim) { return ((scalar_t*)&p)[dim]; }
      inline static __host__ __device__
      scalar_t& get(point_t &p, int dim) { return ((scalar_t*)&p)[dim]; }
    };

    /*! Extract a query_point_t from the node_point_t.
     * The dimension of the query_point_t must be no larger than the dimension
     * of the node_point_t.
     * For example if the node_point_t is a float4 and the query_point_t is a
     * float3, then it returns a float3 with the first 3 dimensions of the float4.
     */
    template<
      typename query_point_t = float4,
      typename node_point_t = float4,
      typename scalar_t = float,
      typename QueryPointInterface = TrivialPointInterface<query_point_t,scalar_t>,
      typename NodePointInterface = TrivialPointInterface<node_point_t,scalar_t>>
    inline __host__ __device__
    query_point_t extractQueryDim(node_point_t node_pt) {
      static_assert(
        point_traits<query_point_t>::numDims <=
          point_traits<node_point_t>::numDims,
        "dimension of query point must be smaller than dimension of node point");
      query_point_t res;
      for(int i=0; i<point_traits<query_point_t>::numDims; ++i) {
        QueryPointInterface::get(res, i) = NodePointInterface::get(node_pt, i);
      }
      return res;
    }

  } // ::cukd::common
} // ::cukd

#define CUKD_CUDA_CHECK( call )                                         \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if (rc != cudaSuccess) {                                            \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      throw std::runtime_error("fatal cuda error");                     \
    }                                                                   \
  }

#define CUKD_CUDA_CALL(call) CUKD_CUDA_CHECK(cuda##call)

#define CUKD_CUDA_CHECK2( where, call )                                 \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if(rc != cudaSuccess) {                                             \
      if (where)                                                        \
        fprintf(stderr, "at %s: CUDA call (%s) "                        \
                "failed with code %d (line %d): %s\n",                  \
                where,#call, rc, __LINE__, cudaGetErrorString(rc));     \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      throw std::runtime_error("fatal cuda error");                     \
    }                                                                   \
  }

#define CUKD_CUDA_SYNC_CHECK()                                  \
  {                                                             \
    cudaDeviceSynchronize();                                    \
    cudaError_t rc = cudaGetLastError();                        \
    if (rc != cudaSuccess) {                                    \
      fprintf(stderr, "error (%s: line %d): %s\n",              \
              __FILE__, __LINE__, cudaGetErrorString(rc));      \
      throw std::runtime_error("fatal cuda error");             \
    }                                                           \
  }



#define CUKD_CUDA_CHECK_NOTHROW( call )                                 \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if (rc != cudaSuccess) {                                            \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      exit(2);                                                          \
    }                                                                   \
  }

#define CUKD_CUDA_CALL_NOTHROW(call) CUKD_CUDA_CHECK_NOTHROW(cuda##call)

#define CUKD_CUDA_CHECK2_NOTHROW( where, call )                         \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if(rc != cudaSuccess) {                                             \
      if (where)                                                        \
        fprintf(stderr, "at %s: CUDA call (%s) "                        \
                "failed with code %d (line %d): %s\n",                  \
                where,#call, rc, __LINE__, cudaGetErrorString(rc));     \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      exit(2);                                                          \
    }                                                                   \
  }

