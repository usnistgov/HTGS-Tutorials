//
// Created by tjb3 on 11/23/15.
//

#ifndef HTGS_PCIAMTASK_H
#define HTGS_PCIAMTASK_H


#include <htgs/api/ICudaTask.hpp>
#include "../data/PCIAMData.h"
#include "../data/CCFData.h"


class PCIAMTask: public htgs::ICudaTask<PCIAMData, CCFData> {

 public:

  ~PCIAMTask() { }

  virtual void initializeCudaGPU(CUcontext context, CUstream stream, int cudaId, int numGPUs, int pipelineId,
                                 int numPipelines);

  virtual void executeGPUTask(std::shared_ptr<PCIAMData> data, CUstream stream);

  virtual void shutdownCuda();

  virtual void debug();

  PCIAMTask(CUcontext *contexts, int *cudaIds, int numGpus, ImageStitching::CUDAImageTile *tile) : ICudaTask(contexts,
                                                                                                             cudaIds,
                                                                                                             numGpus) {
    this->tile = tile;
  }

  virtual std::string getName() override;

  virtual htgs::ITask<PCIAMData, CCFData> *copy() override;

  virtual bool isTerminated(std::shared_ptr<htgs::BaseConnector> inputConnector) override;

 private:
  ImageStitching::CUDATileWorkerMemory *memory;
  ImageStitching::CUDAImageTile *tile;
//    double *pcmMemory;
  cuda_t *originDevMem;
  cuda_t *neighborDevMem;
  int pipelineId;
  int gpuId;
};


#endif //HTGS_PCIAMTASK_H
