//
// Created by tjb3 on 11/18/15.
//

#ifndef HTGS_PCIAMDATA_H
#define HTGS_PCIAMDATA_H

#include <fftw-image-tile.h>
#include "FFTData.h"

enum PCIAMDirection {
  PCIAM_DIRECTION_NORTH = 0,
  PCIAM_DIRECTION_WEST = 1
};

class PCIAMData: public htgs::IData {


 public:
  PCIAMData(std::shared_ptr<FFTData> origin, std::shared_ptr<FFTData> neighbor, PCIAMDirection direction, int order)
      : IData(order) {
    this->origin = origin;
    this->neighbor = neighbor;
    this->direction = direction;
  }

  ~PCIAMData() {

  }

  std::shared_ptr<FFTData> getOrigin() const {
    return this->origin;
  }

  std::shared_ptr<FFTData> getNeighbor() const {
    return this->neighbor;
  }

  PCIAMDirection getDirection() const {
    return this->direction;
  }


 private:
  std::shared_ptr<FFTData> origin;
  std::shared_ptr<FFTData> neighbor;
  PCIAMDirection direction;

};


#endif //HTGS_PCIAMDATA_H
