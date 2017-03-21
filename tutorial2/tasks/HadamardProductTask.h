
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 2/23/16.
//


#ifndef HTGS_HADAMARDPRODUCTTASKWITHMEMEDGE_H
#define HTGS_HADAMARDPRODUCTTASKWITHMEMEDGE_H

#include "../../tutorial-utils/matrix-library/data/MatrixBlockMulData.h"
#include "../../tutorial-utils/matrix-library/data/MatrixBlockData.h"
#include <htgs/api/ITask.hpp>

class HadamardProductTaskWithMemEdge : public htgs::ITask<MatrixBlockMulData<htgs::m_data_t<double>>, MatrixBlockData<htgs::m_data_t<double>>> {

 public:
  HadamardProductTaskWithMemEdge(size_t numThreads) : ITask(numThreads) {}

  virtual ~HadamardProductTaskWithMemEdge() { }

  virtual void executeTask(std::shared_ptr<MatrixBlockMulData<htgs::m_data_t<double>>> data) {

    auto matAData = data->getMatrixA();
    auto matBData = data->getMatrixB();

    htgs::m_data_t<double> matrixA = matAData->getMatrixData();
    htgs::m_data_t<double> matrixB = matBData->getMatrixData();


    size_t width = matAData->getMatrixWidth();
    size_t height = matAData->getMatrixHeight();

    htgs::m_data_t<double> result = this->getMemory<double>("result", new MatrixMemoryRule(1));

    for (size_t i = 0; i < matAData->getMatrixWidth() * matAData->getMatrixHeight(); i++) {
      result->get()[i] = matrixA->get(i) * matrixB->get(i);
    }

    auto matRequest = matAData->getRequest();

    std::shared_ptr<MatrixRequestData>
        matReq(new MatrixRequestData(matRequest->getRow(), matRequest->getCol(), MatrixType::MatrixC));

    addResult(new MatrixBlockData<htgs::m_data_t<double>>(matReq, result, width, height));

    this->releaseMemory(matrixA);
    this->releaseMemory(matrixB);

  }
  virtual std::string getName() {
    return "HadamardProductTaskWithoutMemory";
  }
  virtual HadamardProductTaskWithMemEdge *copy() {
    return new HadamardProductTaskWithMemEdge(this->getNumThreads());
  }

};

#endif //HTGS_HADAMARDPRODUCTTASKWITHMEMEDGE_H
