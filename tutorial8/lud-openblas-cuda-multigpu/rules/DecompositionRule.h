//
// Created by tjb3 on 8/12/16.
//

#ifndef HTGS_TUTORIALS_DECOMPOSITIONRULE_H
#define HTGS_TUTORIALS_DECOMPOSITIONRULE_H

#include <htgs/api/IRule.hpp>
#include "../data/MatrixBlockData.h"
#include "../data/MatrixPanelData.h"
class DecompositionRule : public htgs::IRule<MatrixPanelData, MatrixPanelData>
{


 public:

  DecompositionRule(int numBlocksWidth, int numGpus) :
      numBlocksWidth(numBlocksWidth), numGpus(numGpus)
  {

  }

  virtual void applyRule(std::shared_ptr<MatrixPanelData> data, size_t pipelineId) override {

    // Send the all factored panel to ALL gpus (broadcast)
    if (data->getPanelState() == PanelState::ALL_FACTORED)
    {
      addResult(data->copy());
    }
    else
    {
      // The columns for update start at 1, so convert to 0-based for pipelineId
      long column = data->getPanelCol() - 1;

      int gpuId = (int) column % numGpus;
      if (gpuId == pipelineId)
      {
        addResult(data);
      }

    }


  }
  std::string getName() {
    return "DecompositionRule";
  }

 private:
  int numBlocksWidth;
  long numGpus;

};

#endif //HTGS_TUTORIALS_DECOMPOSITIONRULE_H
