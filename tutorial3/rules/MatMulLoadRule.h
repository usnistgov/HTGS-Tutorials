
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 2/23/16.
//

#ifndef HTGS_MATMULLOADRULE_H
#define HTGS_MATMULLOADRULE_H
#include <htgs/api/IRule.hpp>
#include "../../tutorial-utils/matrix-library/data/MatrixBlockData.h"
#include "../../tutorial-utils/matrix-library/data/MatrixBlockMulData.h"

enum class MatrixState {
  NONE,
  IN_FLIGHT
};

template <class Type>
class MatMulLoadRule : public htgs::IRule<MatrixBlockData<Type>, MatrixBlockMulData<Type>> {

 public:
  MatMulLoadRule (size_t blockWidthA, size_t blockHeightA, size_t blockWidthB, size_t blockHeightB) :
      blockHeightA(blockHeightA), blockWidthA(blockWidthA), blockHeightB(blockHeightB), blockWidthB(blockWidthB) {
    for (size_t i = 0; i < blockWidthA; i++) {
      auto cState = this->allocStateContainer(blockHeightB, blockWidthA, MatrixState::NONE);
      this->matrixCState.push_back(cState);
    }

    this->matrixAState = this->allocStateContainer(blockHeightA, blockWidthA);
    this->matrixBState = this->allocStateContainer(blockHeightB, blockWidthB);
  }

  ~MatMulLoadRule () {
    delete matrixAState;
    delete matrixBState;

    for (auto state : this->matrixCState) {
      delete state;
    }
  }

  void applyRule(std::shared_ptr<MatrixBlockData<Type>> data, size_t pipelineId) {
    std::shared_ptr<MatrixRequestData> request = data->getRequest();

//    std::cout << "Received: " << request->getRow() << ", " << request->getCol() << " "
//              << matrixTypeToString(request->getType()) << std::endl;

    size_t rowA, rowB, colA, colB;
    switch (request->getType()) {

      case MatrixType::MatrixA:
        rowA = request->getRow();
        colA = request->getCol();
        this->matrixAState->set(request->getRow(), request->getCol(), data);

        // B k , j
        rowB = colA;
        for (colB = 0; colB < blockWidthB; colB++) {
          if (this->matrixBState->has(rowB, colB)) {
            auto container = matrixCState[rowB];

            if (!container->has(rowA, colB)) {
//              std::cout << "Sending A(" << rowA << ", " << colA << "); B(" << rowB << ", " << colB << ")"
//                        << "Updating C[" << rowB << "](" << rowA << ", " << colB << ")" << std::endl;
              // Schedule work
              this->addResult(new MatrixBlockMulData<Type>(data, matrixBState->get(rowB, colB), nullptr));
              MatrixState state = MatrixState::IN_FLIGHT;
              container->set(rowA, colB, state);
            }
          }
        }

        break;
      case MatrixType::MatrixB:
        rowB = request->getRow();
        colB = request->getCol();

        this->matrixBState->set(rowB, colB, data);

        colA = rowB;

        for (rowA = 0; rowA < blockHeightA; rowA++) {
          if (this->matrixAState->has(rowA, colA)) {
            auto container = matrixCState[colA];

            if (!container->has(rowA, colB)) {
//              std::cout << "Sending A(" << rowA << ", " << colA << "); B(" << rowB << ", " << colB << ")"
//                        << "Updating C[" << colA << "](" << rowA << ", " << colB << ")" << std::endl;
              // Schedule work
              this->addResult(new MatrixBlockMulData<Type>(matrixAState->get(rowA, colA), data, nullptr));
              MatrixState state = MatrixState::IN_FLIGHT;
              container->set(rowA, colB, state);
            }
          }
        }

        break;
      case MatrixType::MatrixC:break;
      case MatrixType::MatrixAny:break;
    }

//    std::cout << "---------- MATRIX A ---------------" << std::endl;
//    printMatrixA();
//    std::cout << "-----------------------------------" << std::endl;
//    std::cout << "---------- MATRIX B ---------------" << std::endl;
//    printMatrixB();
//    std::cout << "-----------------------------------" << std::endl;
//
//    for (int i = 0; i < blockWidthA; i++)
//    {
//      std::cout << "---------- MATRIX C (" << i << ") -----------" << std::endl;
//      printMatrixC(i);
//      std::cout << "-----------------------------------" << std::endl;
//    }
  }

  void printMatrixA() {
    for (size_t r = 0; r < blockHeightA; r++) {
      for (size_t c = 0; c < blockWidthA; c++) {
        if (matrixAState->has(r, c)) {
          std::cout << "1";
        }
        else {
          std::cout << "0";
        }

      }
      std::cout << std::endl;
    }

  }

  void printMatrixB() {
    for (size_t r = 0; r < blockHeightB; r++) {
      for (size_t c = 0; c < blockWidthB; c++) {
        if (matrixBState->has(r, c)) {
          std::cout << "1";
        }
        else {
          std::cout << "0";
        }

      }
      std::cout << std::endl;
    }
  }

  void printMatrixC(int index) {
    for (size_t r = 0; r < blockHeightA; r++) {
      for (size_t c = 0; c < blockWidthB; c++) {
        if (matrixCState[index]->has(r, c)) {
          std::cout << "1";
        }
        else {
          std::cout << "0";
        }

      }
      std::cout << std::endl;
    }
  }

  std::string getName() {
    return "MatMulLoadRule";
  }

 private:
  size_t blockWidthA;
  size_t blockHeightA;
  size_t blockWidthB;
  size_t blockHeightB;
  htgs::StateContainer<std::shared_ptr<MatrixBlockData<Type>>> *matrixAState;
  htgs::StateContainer<std::shared_ptr<MatrixBlockData<Type>>> *matrixBState;
  std::vector<htgs::StateContainer<MatrixState> *> matrixCState;
};
#endif //HTGS_MATMULLOADRULE_H
