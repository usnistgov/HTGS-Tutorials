
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

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
                   CUcontext *contexts, int *cudaIds, int numGpus, long leadingDimensionFullMatrix) :
      ICudaTask(contexts, cudaIds, numGpus),
      name(name),
      releaseCount(releaseCount),
      blockSize(blockSize),
      leadingDimensionFullMatrix(leadingDimensionFullMatrix) {}

  virtual void
  initializeCudaGPU(CUcontext context, CUstream stream, int cudaId, int numGPUs, int pipelineId, int numPipelines) {
  }

  virtual void executeGPUTask(std::shared_ptr<MatrixBlockData<double *>> data, CUstream stream) {
    std::string matrixName = matrixTypeToString(data->getRequest()->getType());

    // CPU Memory
    double *memoryIn = data->getMatrixData();

    // Cuda Memory
    auto memoryOut = this->memGet<double *>(matrixName + "Copy", new MatrixMemoryRule(releaseCount));

    cublasSetMatrixAsync((int) data->getMatrixHeight(), (int) data->getMatrixWidth(), sizeof(double),
                         memoryIn, (int) leadingDimensionFullMatrix,
                         memoryOut->get(), (int) data->getMatrixHeight(), stream);

    this->syncStream();

    this->addResult(new MatrixBlockData<MatrixMemoryData_t>(data->getRequest(),
                                                            memoryOut,
                                                            data->getMatrixWidth(),
                                                            data->getMatrixHeight()));
  }

  virtual void shutdownCuda() {
  }

  virtual std::string getName() {
    return "CudaCopyInTask(" + name + ")";
  }

  virtual MatrixCopyInTask *copy() {
    return new MatrixCopyInTask(this->name,
                                this->blockSize,
                                this->releaseCount,
                                this->getContexts(),
                                this->getCudaIds(),
                                this->getNumGPUs(),
                                this->leadingDimensionFullMatrix);
  }

 private:
  std::string name;
  int releaseCount;
  double *gpuMemPinned;
  double *scratchSpace;
  int blockSize;
  long leadingDimensionFullMatrix;
};

#endif //HTGS_TUTORIALS_MATRIXCOPYINTASK_H
