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

#pragma once

#include "cukd/common.h"
#include "cukd/helpers.h"
#include "cukd/fcp.h"

namespace cukd {

  /*! runs a k-nearest neighbor operation that tries to fill the
    'currentlyClosest' candidate list (using the number of elemnt k
    and max radius as provided by this class), using the provided
    tree d_nodes with N points. The d_nodes array must be in
    left-balanced kd-tree order. After this class the candidate list
    will contain the k nearest elemnets; if less than k elements
    were found some of the entries in the results list may point to
    a point ID of -1. Return value of the function is the _square_
    of the maximum distance among the k closest elements, if at k
    were found; or the _square_ of the max search radius provided
    for the query */
  template<
    typename math_point_traits_t,
    typename node_point_traits_t=math_point_traits_t,
    typename CandidateList
    >
  inline __device__
  float knn2(CandidateList &currentlyClosest,
            typename math_point_traits_t::point_t queryPoint,
            const typename node_point_traits_t::point_t *d_nodes,
            int N)
  {
    using point_t = typename math_point_traits_t::point_t;
    float maxRadius2 = currentlyClosest.maxRadius2();

    int prev = -1;
    int curr = 0;

    while (true) {
      const int parent = (curr+1)/2-1;
      if (curr >= N) {
        // in some (rare) cases it's possible that below traversal
        // logic will go to a "close child", but may actually only
        // have a far child. In that case it's easiest to fix this
        // right here, pretend we've done that (non-existent) close
        // child, and let parent pick up traversal as if it had been
        // done.
        prev = curr;
        curr = parent;

        continue;
      }
      const int  child = 2*curr+1;
      const bool from_child = (prev >= child);
      if (!from_child) {
        float dist2 = sqrDistance<math_point_traits_t>(queryPoint,d_nodes[curr]);
        if (dist2 <= maxRadius2) {
          currentlyClosest.push(dist2,curr);
          maxRadius2 = currentlyClosest.maxRadius2();
        }
      }

      const auto curr_node = d_nodes[curr];
      int   curr_dim = BinaryTree::levelOf(curr) % DIMENSION;
      float curLoc = queryPoint[curr_dim];
      float curAlt = d_nodes[curr].x[curr_dim];
      const float curr_dim_dist = curLoc-curAlt; //queryPoint[curr_dim];// - curr_node[curr_dim];
      const int   curr_side = curr_dim_dist > 0.f;
      const int   curr_close_child = 2*curr + 1 + curr_side;
      const int   curr_far_child   = 2*curr + 2 - curr_side;

      int next = -1;
      if (prev == curr_close_child)
        // if we came from the close child, we may still have to check
        // the far side - but only if this exists, and if far half of
        // current space if even within search radius.
        next
          = ((curr_far_child<N) && ((curr_dim_dist*curr_dim_dist) <= maxRadius2))
          ? curr_far_child
          : parent;
      else if (prev == curr_far_child)
        // if we did come from the far child, then both children are
        // done, and we can only go up.
        next = parent;
      else
        // we didn't come from any child, so must be coming from a
        // parent... we've already been processed ourselves just now,
        // so next stop is to look at the children (unless there
        // aren't any). this still leaves the case that we might have
        // a child, but only a far child, and this far child may or
        // may not be in range ... we'll fix that by just going to
        // near child _even if_ only the far child exists, and have
        // that child do a dummy traversal of that missing child, then
        // pick up on the far-child logic when we return.
        next
          = (child<N)
          ? curr_close_child
          : parent;

      if (next == -1)
        // if (curr == 0 && from_child)
        // this can only (and will) happen if and only if we come from a
        // child, arrive at the root, and decide to go to the parent of
        // the root ... while means we're done.
        return maxRadius2;
    
      prev = curr;
      curr = next;
    }
  }
  
} // ::cukd

