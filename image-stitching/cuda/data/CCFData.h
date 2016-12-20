//
// Created by tjb3 on 12/2/15.
//

#ifndef HTGS_CCFDATA_H
#define HTGS_CCFDATA_H

#include <htgs/api/IData.hpp>
#include "FFTData.h"
#include "PCIAMData.h"

class CCFData: public htgs::IData {

 public:
  CCFData(int *indices, std::shared_ptr<PCIAMData> pciamData) : IData(pciamData->getOrder()), indices(indices),
                                                                pciamData(pciamData) { }

  ~CCFData() { }


  int *getIndices() const {
    return indices;
  }

  std::shared_ptr<FFTData> getOrigin() const {
    return pciamData->getOrigin();
  }

  std::shared_ptr<FFTData> getNeighbor() const {
    return pciamData->getNeighbor();
  }

  PCIAMDirection getDirection() const {
    return pciamData->getDirection();
  }

 private:
  int *indices;
  std::shared_ptr<PCIAMData> pciamData;

};

#endif //HTGS_CCFDATA_H
