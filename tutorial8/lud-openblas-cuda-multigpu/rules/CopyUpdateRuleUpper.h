//
// Created by tjb3 on 7/29/16.
//

#ifndef HTGS_TUTORIALS_COPYUPDATERULEUPPER_H
#define HTGS_TUTORIALS_COPYUPDATERULEUPPER_H

#include <htgs/api/IRule.hpp>
#include "../data/MatrixFactorData.h"
#include "../../../tutorial-utils/util-matrix.h"
#include "../data/MatrixPanelData.h"
class CopyUpdateRuleUpper : public htgs::IRule<MatrixPanelData, MatrixPanelData>
{

 public:

  CopyUpdateRuleUpper(int windowSize) :
      windowSize(windowSize)
  {
  }

  virtual ~CopyUpdateRuleUpper() {
  }

  virtual void applyRule(std::shared_ptr<MatrixPanelData> data, size_t pipelineId) override {
    int panelCol = data->getPanelCol();
    int activeDiagonal = data->getPanelOperatingDiagonal();
    if (data->getPanelState() != PanelState::ALL_FACTORED && (panelCol - activeDiagonal) > windowSize) {
      addResult(data);
    }
  }

  std::string getName() {
    return "CopyUpdateRuleUpper";
  }
 private:
  int windowSize;

};


#endif //HTGS_TUTORIALS_COPYUPDATERULEUPPER_H
