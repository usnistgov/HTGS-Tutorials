//
// Created by tjb3 on 6/15/16.
//

#ifndef HTGS_TUTORIALS_MATRIXCOPYINTASK_H
#define HTGS_TUTORIALS_MATRIXCOPYINTASK_H

#include <htgs/api/ICudaTask.hpp>
#include "../data/MatrixBlockData.h"
#include <cuda.h>


class MatrixCopyInTask : public htgs::ICudaTask<MatrixBlockData<double *>, MatrixBlockData<MatrixMemoryData_t>> {
 public:
  MatrixCopyInTask(std::string name, int blockSize, int releaseCount,
                   CUcontext *contexts, int *cudaIds, int numGpus, long fullMatrixWidth) :
      ICudaTask(contexts, cudaIds, numGpus),
      name(name), releaseCount(releaseCount), blockSize(blockSize), fullMatrixWidth(fullMatrixWidth)
  {}

  virtual void
  initializeCudaGPU(CUcontext context, CUstream stream, int cudaId, int numGPUs, int pipelineId, int numPipelines) {
    cudaMallocHost((void **)&gpuMemPinned, sizeof(double)*blockSize*blockSize);
    scratchSpace = new double[blockSize*blockSize];
  }

  virtual void executeGPUTask(std::shared_ptr<MatrixBlockData<double *>> data, CUstream stream) {
    std::string matrixName = matrixTypeToString(data->getRequest()->getType());

    // CPU Memory
    double *memoryIn = data->getMatrixData();

    for (int r = 0; r < data->getMatrixHeight(); r++)
    {
      for (int c = 0; c < data->getMatrixWidth(); c++)
      {
        scratchSpace[r * data->getMatrixWidth()+c] = memoryIn[r*fullMatrixWidth+c];
      }
    }


    // Cuda Memory
    auto memoryOut = this->memGet<double *>(matrixName + "Copy", new MatrixMemoryRule(releaseCount));

//    cudaMemcpy(memoryOut->get(), memoryIn->get(), sizeof(double)*data->getMatrixWidth()*data->getMatrixHeight(), cudaMemcpyHostToDevice);

    // TODO Need to copy async 2D

    gpuErrorChk(cudaMemcpyAsync(gpuMemPinned, scratchSpace, sizeof(double) * data->getMatrixHeight()*data->getMatrixWidth(), cudaMemcpyHostToDevice, stream));
//    gpuErrorChk(cudaMemcpy2DAsync(gpuMemPinned, data->getMatrixWidth()*sizeof(double),
//                                  memoryIn, fullMatrixWidth*sizeof(double),
//                                  data->getMatrixWidth()*sizeof(double),
//                                  data->getMatrixHeight(), cudaMemcpyHostToDevice, stream));
    gpuErrorChk(cudaMemcpyAsync(memoryOut->get(), gpuMemPinned, sizeof(double) * data->getMatrixHeight()*data->getMatrixWidth(), cudaMemcpyDeviceToDevice, stream));

    this->syncStream();

    this->addResult(new MatrixBlockData<MatrixMemoryData_t>(data->getRequest(), memoryOut, data->getMatrixWidth(), data->getMatrixHeight()));
  }

  virtual void shutdownCuda() {
    cudaFreeHost(gpuMemPinned);
    delete [] scratchSpace;
  }

  virtual std::string getName() {
    return "CudaCopyInTask(" + name +")";
  }

  virtual MatrixCopyInTask *copy() {
    return new MatrixCopyInTask(this->name, this->blockSize, this->releaseCount, this->getContexts(), this->getCudaIds(), this->getNumGPUs(), this->fullMatrixWidth);
  }

 private:
  std::string name;
  int releaseCount;
  double *gpuMemPinned;
  double *scratchSpace;
  int blockSize;
  long fullMatrixWidth;
};

#endif //HTGS_TUTORIALS_MATRIXCOPYINTASK_H
