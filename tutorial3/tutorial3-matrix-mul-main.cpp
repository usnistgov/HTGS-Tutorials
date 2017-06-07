
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 2/23/16.
//
//#define DEBUG_FLAG
//#define DEBUG_LEVEL_VERBOSE
#include <htgs/api/TaskGraphConf.hpp>
#include <htgs/api/TaskGraphRuntime.hpp>

#include "../tutorial-utils/SimpleClock.h"
#include "../tutorial-utils/util-matrix.h"
#include "../tutorial-utils/matrix-library/operations/matmul.h"
#include "../tutorial-utils/matrix-library/args/MatMulArgs.h"
#include "../tutorial-utils/matrix-library/tasks/LoadMatrixTask.h"
#include "tasks/MatMulBlkTask.h"
#include "tasks/MatMulAccumTask.h"
#include "tasks/MatMulOutputTask.h"
#include "rules/MatMulDistributeRule.h"
#include "rules/MatMulLoadRule.h"
#include "rules/MatMulAccumulateRule.h"

int validateResults(double *matrixC, double *matrixC_HTGS, size_t fullMatrixAHeight, size_t fullMatrixBWidth) {

  if (!validateMatMulResults(20, matrixC, matrixC_HTGS, fullMatrixAHeight*fullMatrixBWidth))
  {
    return -1;
  }

  return 0;
}

void computeSequentialMatMul(double *matrixA, double *matrixB, double *matrixC,
                             size_t fullMatrixAHeight, size_t fullMatrixAWidth, size_t fullMatrixBWidth) {

  computeMatMul(fullMatrixAHeight, fullMatrixBWidth, fullMatrixAWidth, 1.0, matrixA, fullMatrixAWidth, matrixB,
                fullMatrixBWidth, 0.0, matrixC, fullMatrixBWidth, false);
}

int main(int argc, char *argv[]) {
  MatMulArgs matMulArgs;
  matMulArgs.processArgs(argc, argv);

  size_t matrixAHeight = matMulArgs.getMatrixAHeight();
  size_t matrixBWidth = matMulArgs.getMatrixBWidth();
  size_t sharedDim = matMulArgs.getSharedDim();

  size_t blockSize = matMulArgs.getBlockSize();
  size_t numReadThreads = matMulArgs.getNumReadThreads();
  size_t numProdThreads = matMulArgs.getNumMatMulThreads();
  size_t numAccumThreads = (size_t) ceil((double)numProdThreads / 2.0);
  std::string directory = matMulArgs.getDirectory();
  std::string outputDirectory = matMulArgs.getOutputDir();
  bool runSequential = matMulArgs.isRunSequential();
  bool validate = matMulArgs.isValidateResults();

  std::string runtimeFileStr("runtimes");

  int numRetry = 1;

  std::ofstream runtimeFile(runtimeFileStr, std::ios::app);
  double *matrixA = new double[matrixAHeight * sharedDim];
  double *matrixB = new double[matrixBWidth * sharedDim];
  double *matrixC = new double[matrixAHeight * matrixBWidth];

  initMatrix(matrixA, sharedDim, matrixAHeight, false);
  initMatrix(matrixB, matrixBWidth, sharedDim, false);

  for (int numTry = 0; numTry < numRetry; numTry++) {
    SimpleClock clk;
    SimpleClock endToEnd;

    if (runSequential) {
      endToEnd.start();
      initMatMul(numProdThreads);

      clk.start();
      computeSequentialMatMul(matrixA, matrixB, matrixC, matrixAHeight, sharedDim, matrixBWidth);
      clk.stopAndIncrement();
      endToEnd.stopAndIncrement();
    }
    else {
      endToEnd.start();
      initMatMul(1);

      LoadMatrixTask *readAMatTask =
          new LoadMatrixTask(matrixA,
                             numReadThreads,
                             MatrixType::MatrixA,
                             blockSize,
                             sharedDim,
                             matrixAHeight,
                             false);

      LoadMatrixTask *readBMatTask =
          new LoadMatrixTask(matrixB,
                             numReadThreads,
                             MatrixType::MatrixB,
                             blockSize,
                             matrixBWidth,
                             sharedDim,
                             false);

      MatMulBlkTask *mmulTask = new MatMulBlkTask(numProdThreads, false);
      MatMulAccumTask *accumTask = new MatMulAccumTask(numAccumThreads, false);

      MatMulOutputTask *outputTask = new MatMulOutputTask(matrixC, matrixBWidth, blockSize, false);

      size_t blkHeightMatB = readBMatTask->getNumBlocksRows();
      size_t blkWidthMatB = readBMatTask->getNumBlocksCols();

      size_t blkHeightMatA = readAMatTask->getNumBlocksRows();
      size_t blkWidthMatA = readAMatTask->getNumBlocksCols();

      MatMulDistributeRule *distributeRuleMatA = new MatMulDistributeRule(MatrixType::MatrixA);
      MatMulDistributeRule *distributeRuleMatB = new MatMulDistributeRule(MatrixType::MatrixB);

      MatMulLoadRule<double *> *loadRule = new MatMulLoadRule<double *>(blkWidthMatA, blkHeightMatA, blkWidthMatB, blkHeightMatB);
      MatMulAccumulateRule<double *> *accumulateRule = new MatMulAccumulateRule<double *>(blkWidthMatB, blkHeightMatA, blkWidthMatA);

      MatMulOutputRule *outputRule = new MatMulOutputRule(blkWidthMatB, blkHeightMatA, blkWidthMatA);

      auto distributeBk = new htgs::Bookkeeper<MatrixRequestData>();
      auto matMulBk = new htgs::Bookkeeper<MatrixBlockData<double *>>();
      auto matAccumBk = new htgs::Bookkeeper<MatrixBlockData<double *>>();

      auto taskGraph = new htgs::TaskGraphConf<MatrixRequestData, MatrixRequestData>();

      taskGraph->setGraphConsumerTask(distributeBk);
      taskGraph->addRuleEdge(distributeBk, distributeRuleMatA, readAMatTask);
      taskGraph->addRuleEdge(distributeBk, distributeRuleMatB, readBMatTask);

      taskGraph->addEdge(readAMatTask, matMulBk);
      taskGraph->addEdge(readBMatTask, matMulBk);

      taskGraph->addRuleEdge(matMulBk, loadRule, mmulTask);

      taskGraph->addEdge(mmulTask, matAccumBk);
      taskGraph->addRuleEdge(matAccumBk, accumulateRule, accumTask);
      taskGraph->addEdge(accumTask, matAccumBk);

      taskGraph->addRuleEdge(matAccumBk, outputRule, outputTask);
      taskGraph->addGraphProducerTask(outputTask);

      htgs::TaskGraphRuntime *runtime = new htgs::TaskGraphRuntime(taskGraph);

      clk.start();

      runtime->executeRuntime();

      for (size_t row = 0; row < blkHeightMatA; row++) {
        for (size_t col = 0; col < blkWidthMatA; col++) {

          MatrixRequestData *matA = new MatrixRequestData(row, col, MatrixType::MatrixA);
          taskGraph->produceData(matA);

        }
      }

      for (size_t col = 0; col < blkWidthMatB; col++) {
        for (size_t row = 0; row < blkHeightMatB; row++) {
          MatrixRequestData *matB = new MatrixRequestData(row, col, MatrixType::MatrixB);
          taskGraph->produceData(matB);

        }
      }

      taskGraph->finishedProducingData();

      while (!taskGraph->isOutputTerminated()) {
        auto data = taskGraph->consumeData();
        if (data != nullptr) {
        }
      }

      runtime->waitForRuntime();


//      taskGraph->writeDotToFile("profile-graph.dot");
      taskGraph->writeDotToFile("profile-all-threads-graph.dot", DOTGEN_FLAG_SHOW_ALL_THREADING);
      taskGraph->writeDotToFile("profile-graph.dot", DOTGEN_COLOR_COMP_TIME);

      clk.stopAndIncrement();

      delete runtime;
      endToEnd.stopAndIncrement();
    }

    if (validate) {
      double *matrixCTest = new double[matrixAHeight * matrixBWidth];
      computeSequentialMatMul(matrixA, matrixB, matrixCTest, matrixAHeight, sharedDim, matrixBWidth);

      int res = validateResults(matrixC, matrixCTest, matrixAHeight, matrixBWidth);
      if (res != 0) {
        std::cout << "Error validating test failed!" << std::endl;
      }
      else
      {
        std::cout << "Test PASSED" << std::endl;
      }

    }

    std::cout << (runSequential ? "sequential" : "htgs") << ", " << numProdThreads
              << ", accum-threads: " << numAccumThreads << ", width-b: " << matrixBWidth << ", height-a: " << matrixAHeight
              << ", shared-dim: " << sharedDim
              << ", blockSize: " << (runSequential ? 0 : blockSize) 
              << ", time:" << clk.getAverageTime(TimeVal::MILLI)
              << ", end-to-end:" << endToEnd.getAverageTime(TimeVal::MILLI)

        << std::endl;

    runtimeFile << (runSequential ? "sequential" : "htgs") << ", " << numProdThreads
                << ", " << numAccumThreads << ", "
                << matrixBWidth << ", " << matrixAHeight
                << ", " << sharedDim << ", " << blockSize << ", " << clk.getAverageTime(TimeVal::MILLI)
                << ", " << endToEnd.getAverageTime(TimeVal::MILLI)
                << std::endl;

  }

  delete[] matrixA;
  delete[] matrixB;
  delete[] matrixC;

}
