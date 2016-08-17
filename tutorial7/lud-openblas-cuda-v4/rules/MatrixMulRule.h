
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 2/23/16.
//

#ifndef HTGS_MATRIXMULRULE_H
#define HTGS_MATRIXMULRULE_H
#include <htgs/api/IRule.hpp>
#include <vector>
#include "../data/MatrixBlockData.h"
#include "../data/MatrixRequestData.h"
#include "../data/MatrixPanelMulData.h"

enum class MatrixState {
  NONE,
  IN_FLIGHT
};

class MatrixMulRule : public htgs::IRule<MatrixPanelData, MatrixPanelMulData> {

 public:
  MatrixMulRule(int gridHeight, int gridWidth) :
      gridWidth(gridWidth), gridHeight(gridHeight)
  {
    factorState = this->allocStateContainer<bool>(gridHeight, gridWidth, false);
    panels = this->allocStateContainer(gridHeight, gridWidth);
  }

  ~MatrixMulRule() {
    delete panels;
    delete factorState;
  }

  void applyRule(std::shared_ptr<MatrixPanelData> data, int pipelineId) {

    int panelColumn =data->getPanelCol();
    int diagonalColumn = data->getPanelOperatingDiagonal();

    // Data received is factored
    if (data->getPanelState() == PanelState::ALL_FACTORED)
    {
      // Store the diagonal panel
      panels->set(panelColumn, panelColumn, data);

      // Update factor state
      for (int row = diagonalColumn; row < gridHeight; row++)
      {
        factorState->assign(row, panelColumn, true);
      }

      // Check panels to see if they are ready to be matmul'd matmul to panels waiting to be updated for this diagonal
      for (int col = panelColumn + 1; col < gridWidth; col++)
      {
        if (factorState->has(diagonalColumn, col))
        {
          auto updatePanel = panels->get(diagonalColumn, col);
          addResult(new MatrixPanelMulData(data, updatePanel));
        }
      }

    }
      // Data received is not factored, except for first block
    else if (data->getPanelState() == PanelState::TOP_FACTORED)
    {
      factorState->assign(diagonalColumn, panelColumn, true);
      panels->set(diagonalColumn, panelColumn, data);

      // Check if the diagonal panel has been factored. If it has then send the matmul
      if (factorState->has(diagonalColumn, diagonalColumn))
      {
        auto factoredPanel = panels->get(diagonalColumn, diagonalColumn);
        addResult(new MatrixPanelMulData(factoredPanel, data));
      }

    }

//    std::cout << "Factor State" << std::endl;
//    factorState->printContents();
//    std::cout << std::endl;

//    std::cout << "Update State" << std::endl;
//    updateState->printContents();
//    std::cout << std::endl;
  }


  std::string getName() {
    return "MatrixMulRule";
  }

 private:

  htgs::StateContainer<std::shared_ptr<MatrixPanelData>> *panels;
  htgs::StateContainer<bool> *factorState;
  int gridWidth;
  int gridHeight;
};
#endif //HTGS_MATRIXMULRULE_H
