//
// Created by tjb3 on 6/22/16.
//

#include "util-cuda.h"
CUcontext *initCuda(int nGPUs, int *gpuIDs) {
  cuInit(0);

  CUcontext *contexts = (CUcontext *) malloc(nGPUs * sizeof(CUcontext));

  for (int i = 0; i < nGPUs; i++) {
    CUdevice device;
    cuDeviceGet(&device, gpuIDs[i]);

    CUcontext context;
    cuCtxCreate_v2(&context, 0, device);
    contexts[i] = context;
  }

  return contexts;
}