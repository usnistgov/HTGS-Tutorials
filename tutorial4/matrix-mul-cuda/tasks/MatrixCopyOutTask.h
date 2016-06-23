
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

class MatrixCopyOutTask : public htgs::ICudaTask<MatrixBlockData<MatrixMemoryData_t>, MatrixBlockData<double *>> {
 public:
  MatrixCopyOutTask(std::string name, int blockSize, CUcontext *contexts, int *cudaIds, int numGpus) : ICudaTask(
      contexts,
      cudaIds,
      numGpus),
                                                                                                       name(name),
                                                                                                       blockSize(
                                                                                                           blockSize) {}

  virtual void
  initializeCudaGPU(CUcontext context, CUstream stream, int cudaId, int numGPUs, int pipelineId, int numPipelines) {
    cudaMallocHost((void **) &cudaMemPinned, sizeof(double) * blockSize * blockSize);
  }

  virtual void executeGPUTask(std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> data, CUstream stream) {
    // Cuda Memory
    auto memoryIn = data->getMatrixData();

    // CPU Memory
    double *memoryOut = new double[data->getMatrixHeight() * data->getMatrixWidth()];

    gpuErrorChk(cudaMemcpyAsync(cudaMemPinned,
                                memoryIn->get(),
                                sizeof(double) * data->getMatrixHeight() * data->getMatrixWidth(),
                                cudaMemcpyDeviceToDevice,
                                stream));
    gpuErrorChk(cudaMemcpyAsync(memoryOut,
                                cudaMemPinned,
                                sizeof(double) * data->getMatrixHeight() * data->getMatrixWidth(),
                                cudaMemcpyDeviceToHost,
                                stream));

    this->syncStream();

    this->memRelease(name, memoryIn);

    this->addResult(new MatrixBlockData<double *>(data->getRequest(),
                                                  memoryOut,
                                                  data->getMatrixWidth(),
                                                  data->getMatrixHeight()));
  }

  virtual void shutdownCuda() {
    cudaFreeHost(cudaMemPinned);
  }

  virtual std::string getName() {
    return "CudaCopyOutTask(" + name + ")";
  }

  virtual htgs::ITask<MatrixBlockData<MatrixMemoryData_t>, MatrixBlockData<double *>> *copy() {
    return new MatrixCopyOutTask(this->name, blockSize, this->getContexts(), this->getCudaIds(), this->getNumGPUs());
  }

 private:
  std::string name;
  int blockSize;
  double *cudaMemPinned;
};

#endif //HTGS_TUTORIALS_MATRIXCOPYOUTTASK_H
