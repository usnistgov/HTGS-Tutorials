
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 6/15/16.
//

#ifndef HTGS_TUTORIALS_MATRIXCOPYOUTTASK_H
#define HTGS_TUTORIALS_MATRIXCOPYOUTTASK_H

#include <htgs/api/ICudaTask.hpp>
#include "../data/MatrixBlockData.h"
#include <cuda.h>
#include <cublas_v2.h>

class MatrixCopyOutTask : public htgs::ICudaTask<MatrixBlockData<MatrixMemoryData_t>, MatrixBlockData<double *>> {
 public:
  MatrixCopyOutTask(int blockSize, CUcontext *contexts, int *cudaIds, int numGpus,
                    htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>> *matrixBlocks, long matrixLeadingDimension) :
      ICudaTask(contexts, cudaIds, numGpus), blockSize(blockSize), matrixBlocks(matrixBlocks), matrixLeadingDimension(matrixLeadingDimension) {}

  virtual void
  initializeCudaGPU(CUcontext context, CUstream stream, int cudaId, int numGPUs, int pipelineId, int numPipelines) {
//    cudaMallocHost((void **)&cudaMemPinned, sizeof(double)*blockSize*blockSize);
  }

  virtual void executeGPUTask(std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> data, CUstream stream) {
    // Cuda Memory
    auto memoryIn = data->getMatrixData();

    // CPU Memory
    auto result = matrixBlocks->get(data->getRequest()->getRow(), data->getRequest()->getCol());
    double *memoryOut = result->getMatrixData();

    cublasGetMatrixAsync((int) data->getMatrixHeight(), (int) data->getMatrixWidth(), sizeof(double),
                         memoryIn->get(), (int) data->getMatrixHeight(),
                         memoryOut, (int) matrixLeadingDimension, stream);

    this->syncStream();

//    MatrixMemoryRule * rule = (MatrixMemoryRule *)memoryIn->getMemoryReleaseRule();
//    std::cout << " Release count for matrix result: " << data->getRequest()->getRow() << ", " << data->getRequest()->getCol() << ":  " << rule->getReleaseCount() << std::endl;

    this->memRelease("ResultMatrixMem", memoryIn);

    this->addResult(result);
  }

  virtual void shutdownCuda() {
  }

  virtual std::string getName() {
    return "CudaCopyOutTask";
  }

  virtual htgs::ITask<MatrixBlockData<MatrixMemoryData_t>, MatrixBlockData<double *>> *copy() {
    return new MatrixCopyOutTask(blockSize, this->getContexts(), this->getCudaIds(), this->getNumGPUs(), matrixBlocks, matrixLeadingDimension);
  }

 private:
  int blockSize;
  htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>> *matrixBlocks;
  long matrixLeadingDimension;
};

#endif //HTGS_TUTORIALS_MATRIXCOPYOUTTASK_H
