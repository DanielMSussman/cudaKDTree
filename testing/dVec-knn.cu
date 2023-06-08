// ======================================================================== //
// Copyright 2022-2022 Ingo Wald                                            //
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

#include "cukd/builder.h"
// knn = "k-nearest-neighbor" query
#include "cukd/knn.h"
#include "cukd/generalStructureKnn.h"
#include "dDimensionalVectorTypes.h"
#include <queue>
#include <limits>
#include <iomanip>

using namespace cukd;
using namespace cukd::common;



dVec *generatePoints(int N)
{
  std::cout << "generating " << N <<  " points" << std::endl;
  dVec *d_points = 0;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_points,N*sizeof(dVec)));
  for (int i=0;i<DIMENSION;i++) {
    for (int dd = 0; dd < DIMENSION; ++dd)
        d_points[i][dd] = (float)drand48();
  }
  return d_points;
}

// ==================================================================
__global__ void d_knn25(scalar *d_results,
                       dVec *d_queries,
                       int numQueries,
                       dVec *d_nodes,
                       int numNodes,
                       scalar maxRadius)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;

  cukd::HeapCandidateList<25> result(maxRadius);
  scalar sqrDist
    = cukd::knn2
    <cukd::TrivialFloatPointTraits<dVec,DIMENSION>>
    (result,d_queries[tid],d_nodes,numNodes);
  d_results[tid] = sqrtf(sqrDist);
}

void knn25(scalar *d_results,
          dVec *d_queries,
          int numQueries,
          dVec *d_nodes,
          int numNodes,
          scalar maxRadius)
{
  int bs = 128;
  int nb = cukd::common::divRoundUp(numQueries,bs);
  d_knn25<<<nb,bs>>>(d_results,d_queries,numQueries,d_nodes,numNodes,maxRadius);
}

inline void verifyKNN(int pointID, int k, float maxRadius,
                      dVec *points, int numPoints,
                      dVec queryPoint,
                      float presumedResult)
{
  std::priority_queue<float> closest_k;
  for (int i=0;i<numPoints;i++) {
    float dist2 = cukd::sqrDistance<cukd::TrivialFloatPointTraits<dVec>>(queryPoint,points[i]);
    float d = sqrtf(dist2);//cukd::distance(queryPoint,points[i]);
    if (d <= maxRadius)
      closest_k.push(d);
    if (closest_k.size() > k)
      closest_k.pop();
  }

  float actualResult = (closest_k.size() == k) ? closest_k.top() : maxRadius;


  // check if the top 21-ish bits are the same; this will allow the
  // compiler to produce slightly different results on host and device
  // (usually caused by it uses madd's on one and separate +/* on
  // t'other...
  bool closeEnough
    =  /* this catches result==inf:*/(actualResult == presumedResult)
    || /* this catches bit errors: */(fabsf(actualResult - presumedResult) <= 1e-6f);
  
  if (!closeEnough) {
    std::cout << "for point #" << pointID << ": "
              << "verify found max dist " << std::setprecision(10) << actualResult
              << " (bits " << (int*)(uint64_t)((uint32_t&)actualResult)
              << "), knn reported " << presumedResult
              << " (bits " << (int*)(uint64_t)((uint32_t&)presumedResult)
              << "), difference is " << (actualResult-presumedResult)
              << std::endl;
    throw std::runtime_error("verification failed");
  }
}

int main(int ac, const char **av)
{
  using namespace cukd::common;


    TrivialFloatPointTraits<float3> tt1;
    TrivialFloatPointTraits<float4> t2;
    TrivialFloatPointTraits<dVec> t3;
    TrivialFloatPointTraits<dVec,DIMENSION> t4;

    std::cout << tt1.numDims << "\t" << t2.numDims << "\t" << t3.numDims << "\t" <<t4.numDims << std::endl;

  int nPoints = 173;
  bool verify = false;
  scalar maxQueryRadius = std::numeric_limits<scalar>::infinity();
  int nRepeats = 1;
  for (int i=1;i<ac;i++) {
    std::string arg = av[i];
    if (arg[0] != '-')
      nPoints = std::stoi(arg);
    else if (arg == "-v")
      verify = true;
    else if (arg == "-nr")
      nRepeats = atoi(av[++i]);
    else if (arg == "-r")
      maxQueryRadius = std::stof(av[++i]);
    else
      throw std::runtime_error("known cmdline arg "+arg);
  }
  if(!verify)
      std::cout << "each point is " << DIMENSION << " dimensional" << std::endl;
  dVec *d_points = generatePoints(nPoints);

  {
    double t0 = getCurrentTime();
    std::cout << "calling builder..." << std::endl;
    cukd::buildTree
      <cukd::TrivialFloatPointTraits<dVec,DIMENSION>>
      (d_points,nPoints);
    CUKD_CUDA_SYNC_CHECK();
    double t1 = getCurrentTime();
    std::cout << "done building tree, took " << prettyDouble(t1-t0) << "s" << std::endl;
  }

  int nQueries = 10000000;
  dVec *d_queries = generatePoints(nQueries);
  scalar  *d_results;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_results,nQueries*sizeof(scalar)));

    std::cout << "running " << nRepeats << " sets of knn25 queries..." << std::endl;
    double t0 = getCurrentTime();
    for (int i=0;i<nRepeats;i++)
      knn25(d_results,d_queries,nQueries,d_points,nPoints,maxQueryRadius);
    CUKD_CUDA_SYNC_CHECK();
    double t1 = getCurrentTime();
    std::cout << "done " << nRepeats << " iterations of knn25 query, took " << prettyDouble(t1-t0) << "s" << std::endl;
    std::cout << " that's " << prettyDouble((t1-t0)/nRepeats) << "s per complete query set (avg)..." << std::endl;
    std::cout << " ... or " << prettyDouble(nQueries*nRepeats/(t1-t0)) << " queries/s" << std::endl;

    if (verify) {
      std::cout << "verifying result ..." << std::endl;
      for (int i=0;i<nQueries;i++)
        verifyKNN(i,25,maxQueryRadius,d_points,nPoints,d_queries[i],d_results[i]);
      std::cout << "verification passed ... " << std::endl;
    }

}
