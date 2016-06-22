//
// Created by tjb3 on 6/22/16.
//

#ifndef HTGS_TUTORIALS_UTIL_CUDA_H
#define HTGS_TUTORIALS_UTIL_CUDA_H

#include <cuda.h>
CUcontext *initCuda(int nGPUs, int *gpuIDs);

#endif //HTGS_TUTORIALS_UTIL_CUDA_H
