//
// Created by tjb3 on 11/23/15.
//

#include "PCIAMTask.h"
#include <fftw-stitching.h>
#include <htgs/api/ITask.hpp>

void PCIAMTask::shutdown() {
  delete this->memory;
}

void PCIAMTask::executeTask(std::shared_ptr<PCIAMData> data) {
  ImageStitching::FFTWImageTile *origin = data->getOrigin()->getTile();
  ImageStitching::FFTWImageTile *neighbor = data->getNeighbor()->getTile();

  PCIAMDirection direction = data->getDirection();

  ImageStitching::CorrelationTriple triple = phaseCorrelationImageAlignmentFFTW(neighbor, origin, memory);


  switch (direction) {
    case PCIAM_DIRECTION_NORTH:
      origin->setNorthTranslation(triple);
//            std::cout << "North: " << origin->getFilename() << "-> " << neighbor->getFilename() << " " << triple << std::endl;
      break;
    case PCIAM_DIRECTION_WEST:
      origin->setWestTranslation(triple);
//            std::cout << "West: " << std::string(origin->getFilename()) << "-> " << std::string(neighbor->getFilename()) << " " << triple << std::endl;
      break;
  }

  // release memory
  if (data->getOrigin()->getReadMemory() != nullptr)
    this->releaseMemory(data->getOrigin()->getReadMemory());
  if (data->getNeighbor()->getReadMemory() != nullptr)
    this->releaseMemory(data->getNeighbor()->getReadMemory());

  if (data->getOrigin()->getFFTMemory() != nullptr)
    this->releaseMemory(data->getOrigin()->getFFTMemory());
  if (data->getNeighbor()->getFFTMemory() != nullptr)
    this->releaseMemory(data->getNeighbor()->getFFTMemory());

}

std::string PCIAMTask::getName() {
  return "PCIAM Task";
}

htgs::ITask<PCIAMData, PCIAMData> *PCIAMTask::copy() {
  return new PCIAMTask(this->getNumThreads(), this->tile);
}

