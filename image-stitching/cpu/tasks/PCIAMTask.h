//
// Created by tjb3 on 11/23/15.
//

#ifndef HTGS_PCIAMTASK_H
#define HTGS_PCIAMTASK_H


#include <htgs/api/ITask.hpp>
#include "../data/PCIAMData.h"


class PCIAMTask: public htgs::ITask<PCIAMData, PCIAMData> {

 public:

  PCIAMTask(size_t numThreads, ImageStitching::FFTWImageTile *tile) : ITask(numThreads) {
    this->tile = tile;
    this->memory = new ImageStitching::FFTWTileWorkerMemory(this->tile);
  }

  ~PCIAMTask() { }

  virtual void shutdown() override;

  virtual void executeTask(std::shared_ptr<PCIAMData> data) override;

  virtual std::string getName() override;

  virtual htgs::ITask<PCIAMData, PCIAMData> *copy() override;

 private:
  ImageStitching::FFTWTileWorkerMemory *memory;
  ImageStitching::FFTWImageTile *tile;

};


#endif //HTGS_PCIAMTASK_H
