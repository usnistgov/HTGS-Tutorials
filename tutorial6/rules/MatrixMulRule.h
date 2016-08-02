
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
#include "../data/MatrixBlockMulData.h"
#include "../data/MatrixRequestData.h"

enum class MatrixState {
  NONE,
  IN_FLIGHT
};

class MatrixMulRule : public htgs::IRule<MatrixBlockData<double *>, MatrixBlockMulData<double *>> {

 public:
  MatrixMulRule(htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>> *matrixBlocks,
                 int gridHeight, int gridWidth) :
      matrixBlocks(matrixBlocks), gridWidth(gridWidth), gridHeight(gridHeight)
  {
    matrixState = this->allocStateContainer<int>(gridHeight, gridWidth, 0);
    updateState = this->allocStateContainer<int>(gridHeight, gridWidth, 0);
  }

  ~MatrixMulRule() {
    delete matrixState;
    delete updateState;
  }

  void applyRule(std::shared_ptr<MatrixBlockData<double *>> data, int pipelineId) {
    std::shared_ptr<MatrixRequestData> request = data->getRequest();

    int dataRow = request->getRow();
    int dataCol = request->getCol();

    // increment the data received
    int dataState = matrixState->get(dataRow, dataCol) + 1;
    matrixState->set(dataRow, dataCol, dataState);

    // Check if matrix is above the diagonal
    if (dataCol > dataRow)
    {
      // Check if rows are ready
      for (int r = dataRow+1; r < gridHeight; r++)
      {
        // row = data col (below the diagonal)
        int stateTest = matrixState->get(r, dataRow);

        int resultTest = updateState->get(r, dataCol);

        // If the data state matches the state test, then initiate the matmul
        if (stateTest == dataState && resultTest == stateTest-1)
        {
          // data is matrixB
          auto matrixA = this->matrixBlocks->get(r, dataRow);

          auto matrixResult = this->matrixBlocks->get(r, dataCol);
          addResult(new MatrixBlockMulData<double *>(matrixA, data, matrixResult));

          int uState = updateState->get(r, dataCol) + 1;
          updateState->set(r, dataCol, uState);

        }
      }
    }
      // matrix is below the diagonal
    else if (dataRow > dataCol)
    {
      // Check if cols are ready
      for (int c = dataCol+1; c < gridWidth; c++)
      {
        // col = data row
        int stateTest = matrixState->get(dataCol, c);
        int resultTest = updateState->get(dataRow, c);
        // If the data state matches the state test, then initiate the matmul
        if (stateTest == dataState && resultTest == stateTest-1)
        {
          // data is matrixA
          auto matrixB = this->matrixBlocks->get(dataCol, c);
          auto matrixResult = this->matrixBlocks->get(dataRow, c);
          addResult(new MatrixBlockMulData<double *>(data, matrixB, matrixResult));
          int uState = updateState->get(dataRow, c) + 1;
          updateState->set(dataRow, c, uState);
        }

      }
    }

//    std::cout << "MatMulState" << std::endl;
//    matrixState->printContents();
//    std::cout << std::endl;
//
//    std::cout << "Update Result State" << std::endl;
//    updateState->printContents();
//    std::cout << std::endl;

  }


  std::string getName() {
    return "MatrixMulRule";
  }

 private:
  htgs::StateContainer<int> *matrixState;
  htgs::StateContainer<int> *updateState;
  htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>> *matrixBlocks;
  int gridWidth;
  int gridHeight;
};
#endif //HTGS_MATRIXMULRULE_H
