//
// Created by tjb3 on 7/29/16.
//

#ifndef HTGS_TUTORIALS_COPYUPDATERULEUPPERWINDOW_H
#define HTGS_TUTORIALS_COPYUPDATERULEUPPERWINDOW_H

#include <htgs/api/IRule.hpp>
#include "../data/MatrixFactorData.h"
#include "../../../tutorial-utils/util-matrix.h"
#include "../data/MatrixPanelData.h"
class CopyUpdateRuleUpperWindow : public htgs::IRule<MatrixPanelData, MatrixPanelData>
{

 public:

  CopyUpdateRuleUpperWindow(int windowSize) :
      windowSize(windowSize)
  {
  }

  virtual ~CopyUpdateRuleUpperWindow() {
  }

  virtual void applyRule(std::shared_ptr<MatrixPanelData> data, int pipelineId) {
    int panelCol = data->getPanelCol();
    int activeDiagonal = data->getPanelOperatingDiagonal();
    if (data->getPanelState() != PanelState::ALL_FACTORED && (panelCol - activeDiagonal) <= windowSize) {
      addResult(data);
    }
  }

  std::string getName() {
    return "CopyUpdateRuleUpperWindow";
  }
 private:
  int windowSize;

};


#endif //HTGS_TUTORIALS_COPYUPDATERULEUPPERWINDOW_H
