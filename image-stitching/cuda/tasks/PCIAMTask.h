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

  virtual void initializeCudaGPU() override;

  virtual void executeTask(std::shared_ptr<PCIAMData> data);

  virtual void shutdownCuda();


  PCIAMTask(int *cudaIds, int numGpus, ImageStitching::CUDAImageTile *tile) : ICudaTask(cudaIds, numGpus) {
    this->tile = tile;
  }

  virtual std::string getName() override;

  virtual htgs::ITask<PCIAMData, CCFData> *copy() override;

 private:
  ImageStitching::CUDATileWorkerMemory *memory;
  ImageStitching::CUDAImageTile *tile;
//    double *pcmMemory;
  cuda_t *originDevMem;
  cuda_t *neighborDevMem;
};


#endif //HTGS_PCIAMTASK_H
