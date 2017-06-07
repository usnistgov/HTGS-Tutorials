//
// Created by tjb3 on 11/17/15.
//

#include "ReadTask.h"
#include "../../cpu/memory/FFTMemoryRule.h"


void ReadTask::executeTask(std::shared_ptr<FFTData> data) {

  ImageStitching::CUDAImageTile *tile = data->getTile();
  tile->setThreadId(this->getPipelineId());


  if (this->hasMemoryEdge("read")) {
    std::shared_ptr<htgs::MemoryData<img_t>>
        memory = this->getMemory<img_t>("read", new FFTMemoryRule(tile, this->extentWidth,
                                                                 this->extentHeight,
                                                                 this->startRow, this->startCol));
    tile->readTile(memory->get());
    data->setReadData(memory);
  }
  else {
    tile->readTile();
  }

  if (this->hasMemoryEdge("fft")) {
    std::shared_ptr<htgs::MemoryData<cuda_t>> fftMemory = this->getMemory<cuda_t>("fft",
                                                                                   new FFTMemoryRule(tile,
                                                                                                     this->extentWidth,
                                                                                                     this->extentHeight,
                                                                                                     this->startRow,
                                                                                                     this->startCol));
    data->setFftData(fftMemory);
  }


  this->addResult(data);
}

std::string ReadTask::getName() {
  return "Read Task";
}

htgs::ITask<FFTData, FFTData> *ReadTask::copy() {
  return new ReadTask(this->startCol, this->startRow, this->extentWidth, this->extentHeight);
}
