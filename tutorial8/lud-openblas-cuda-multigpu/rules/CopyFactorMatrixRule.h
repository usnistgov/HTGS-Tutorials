//
// Created by tjb3 on 8/12/16.
//

#ifndef HTGS_TUTORIALS_COPYFACTORMATRIXRULE_H
#define HTGS_TUTORIALS_COPYFACTORMATRIXRULE_H

#include <htgs/api/IRule.hpp>
#include "../data/MatrixBlockData.h"
#include "../data/MatrixPanelData.h"
class CopyFactorMatrixRule : public htgs::IRule<MatrixPanelData, MatrixPanelData>
{


 public:

  CopyFactorMatrixRule()
  { }

  virtual void applyRule(std::shared_ptr<MatrixPanelData> data, int pipelineId) {
    // Send the all factored panel to ALL gpus (broadcast)
    if (data->getPanelState() == PanelState::ALL_FACTORED)
    {
      addResult(data);
    }
  }
  std::string getName() {
    return "CopyFactorMatrixRule";
  }

 private:

};

#endif //HTGS_TUTORIALS_COPYFACTORMATRIXRULE_H
