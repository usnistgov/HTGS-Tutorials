//
// Created by tjb3 on 2/23/16.
//
//#define DEBUG_FLAG
//#define DEBUG_LEVEL_VERBOSE
#include <htgs/api/TaskGraph.hpp>
#include <htgs/api/Runtime.hpp>
#include <cblas.h>

#include "data/MatrixRequestData.h"
#include "data/MatrixBlockData.h"
#include "tasks/ReadMatrixTask.h"
#include "rules/MatrixLoadRule.h"
#include "tasks/MatrixMulBlkTask.h"
#include "tasks/MatrixAccumTask.h"
#include "tasks/OutputTask.h"
#include "rules/MatrixAccumulateRule.h"
#include "rules/MatrixDistributeRule.h"
#include "rules/MatrixOutputRule.h"
#include "../../tutorial-utils/SimpleClock.h"
#include "../../tutorial-utils/util-matrix.h"

int validateResults(double *matrixC, double *matrixC_HTGS, int fullMatrixAHeight, int fullMatrixBWidth) {
  int count = 0;

  for (int i = 0; i < fullMatrixAHeight * fullMatrixBWidth; i++) {
    if (matrixC[i] != matrixC_HTGS[i]) {
      if (count <= 20) {
        std::cout << std::fixed << i << ": cMem = " << matrixC[i] << " cMemPar = " << matrixC_HTGS[i] << std::endl;
      }
      count++;
    }
  }

  if (count > 0)
    std::cout << "Total incorrect = " << count << std::endl;

  if (count > 0)
    return 1;
  return 0;
}

void computeSequentialMatMul(double *matrixA,
                             double *matrixB,
                             double *matrixC,
                             int fullMatrixAHeight,
                             int fullMatrixAWidth,
                             int fullMatrixBWidth) {

  cblas_dgemm(CblasRowMajor,
              CblasNoTrans,
              CblasNoTrans,
              fullMatrixAHeight,
              fullMatrixBWidth,
              fullMatrixAWidth,
              1.0,
              matrixA,
              fullMatrixAWidth,
              matrixB,
              fullMatrixBWidth,
              0.0,
              matrixC,
              fullMatrixBWidth);
}

int main(int argc, char *argv[]) {
  long matrixAHeight = 1024;
  long matrixBWidth = 1024;
  long sharedDim = 1024;

  int blockSize = 512;
  int numReadThreads = 1;
  int numProdThreads = 26;
  int numBlasThreads = 40;
  bool runSequential = false;
//  bool validate = false;

  std::string runtimeFileStr("runtimes");

  int numRetry = 1;

  if (argc > 1) {
    for (int arg = 1; arg < argc; arg++) {
      std::string argvs(argv[arg]);

      if (argvs == "--width-b") {
        arg++;
        matrixBWidth = atoi(argv[arg]);
      }

      if (argvs == "--height-a") {
        arg++;
        matrixAHeight = atoi(argv[arg]);
      }

      if (argvs == "--shared-dim") {
        arg++;
        sharedDim = atoi(argv[arg]);
      }

      if (argvs == "--run-sequential") {
        runSequential = true;
      }

      if (argvs == "--num-retry" && arg + 1 < argc) {
        arg++;
        numRetry = atoi(argv[arg]);
      }

      if (argvs == "--block-size") {
        arg++;
        blockSize = atoi(argv[arg]);
      }

      if (argvs == "--num-threads-htgs" && arg + 1 < argc) {
        numProdThreads = atoi(argv[arg + 1]);
        arg++;
      }

      if (argvs == "--num-threads-blas" && arg + 1 < argc) {
        numBlasThreads = atoi(argv[arg + 1]);
        arg++;
      }

      if (argvs == "--runtime-file" && arg + 1 < argc) {
        runtimeFileStr = argv[arg + 1];
        arg++;
      }

//      if (argvs == "--validate-results") {
//        validate = true;
//      }

      if (argvs == "--help") {
        std::cout << argv[0]
                  << " args: [--width-b <#>] [--height-a <#>] [--shared-dim <#>] [--block-size <#>] [--num-retry <#>] [--num-threads-htgs <#>] [--num-threads-blas <#>] [--runtime-file <filename>] [--dir <dir>] [--output-dir <dir>] [--validate-results] [--run-sequential] [--help]"
                  << std::endl;
        exit(0);
      }
    }
  }

  std::ofstream runtimeFile(runtimeFileStr, std::ios::app);
  double *matrixA = new double[matrixAHeight * sharedDim];
  double *matrixB = new double[matrixBWidth * sharedDim];
  double *matrixC = new double[matrixAHeight * matrixBWidth];
//  double *matrixC_HTGS = new double[matrixAHeight*matrixBWidth];

  initMatrix(matrixA, sharedDim, matrixAHeight, false);
  initMatrix(matrixB, matrixBWidth, sharedDim, false);

  for (int numTry = 0; numTry < numRetry; numTry++) {
    SimpleClock clk;

    if (runSequential) {
      openblas_set_num_threads(numBlasThreads);

      clk.start();
      computeSequentialMatMul(matrixA, matrixB, matrixC, matrixAHeight, sharedDim, matrixBWidth);
      clk.stopAndIncrement();
    }
    else {

      openblas_set_num_threads(1);

      ReadMatrixTask *readAMatTask =
          new ReadMatrixTask(numReadThreads,
                             MatrixType::MatrixA,
                             blockSize,
                             sharedDim,
                             matrixAHeight,
                             matrixA,
                             "A");
      ReadMatrixTask *readBMatTask =
          new ReadMatrixTask(numReadThreads,
                             MatrixType::MatrixB,
                             blockSize,
                             matrixBWidth,
                             sharedDim,
                             matrixB,
                             "B");
      MatrixMulBlkTask *mmulTask =
          new MatrixMulBlkTask(numProdThreads, sharedDim, matrixAHeight, matrixBWidth, sharedDim, blockSize);
      MatrixAccumTask *accumTask = new MatrixAccumTask((int) ceil((double) numProdThreads / 2.0));

      OutputTask *outputTask = new OutputTask(matrixC, matrixBWidth, matrixAHeight, blockSize);

      int blkHeightMatB = readBMatTask->getNumBlocksRows();
      int blkWidthMatB = readBMatTask->getNumBlocksCols();

      int blkHeightMatA = readAMatTask->getNumBlocksRows();
      int blkWidthMatA = readAMatTask->getNumBlocksCols();

      MatrixDistributeRule *distributeRuleMatA = new MatrixDistributeRule(MatrixType::MatrixA);
      MatrixDistributeRule *distributeRuleMatB = new MatrixDistributeRule(MatrixType::MatrixB);

      MatrixLoadRule *loadRule = new MatrixLoadRule(blkWidthMatA, blkHeightMatA, blkWidthMatB, blkHeightMatB);
      MatrixAccumulateRule *accumulateRule = new MatrixAccumulateRule(blkWidthMatB, blkHeightMatA, blkWidthMatA);

      MatrixOutputRule *outputRule = new MatrixOutputRule(blkWidthMatB, blkHeightMatA, blkWidthMatA);

      auto distributeBk = new htgs::Bookkeeper<MatrixRequestData>();
      auto matMulBk = new htgs::Bookkeeper<MatrixBlockData<double *>>();
      auto matAccumBk = new htgs::Bookkeeper<MatrixBlockData<double *>>();

      auto taskGraph = new htgs::TaskGraph<MatrixRequestData, MatrixBlockData<double *>>();

      taskGraph->addGraphInputConsumer(distributeBk);
      taskGraph->addRule(distributeBk, readAMatTask, distributeRuleMatA);
      taskGraph->addRule(distributeBk, readBMatTask, distributeRuleMatB);

      taskGraph->addEdge(readAMatTask, matMulBk);
      taskGraph->addEdge(readBMatTask, matMulBk);

      taskGraph->addRule(matMulBk, mmulTask, loadRule);

      taskGraph->addEdge(mmulTask, matAccumBk);
      taskGraph->addRule(matAccumBk, accumTask, accumulateRule);
      taskGraph->addEdge(accumTask, matAccumBk);

      taskGraph->addRule(matAccumBk, outputTask, outputRule);
      taskGraph->addGraphOutputProducer(outputTask);

      taskGraph->incrementGraphInputProducer();

      htgs::Runtime *runtime = new htgs::Runtime(taskGraph);

      clk.start();

      runtime->executeRuntime();

      for (int row = 0; row < blkHeightMatA; row++) {
        for (int col = 0; col < blkWidthMatA; col++) {

          MatrixRequestData *matA = new MatrixRequestData(row, col, MatrixType::MatrixA);
          taskGraph->produceData(matA);

        }
      }

      for (int row = 0; row < blkHeightMatB; row++) {
        for (int col = 0; col < blkWidthMatB; col++) {

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
      clk.stopAndIncrement();

      delete runtime;
    }

//    if (validate) {
//      int res = validateResults(matrixC, matrixC_HTGS, matrixAHeight, matrixBWidth);
//      if (res != 0) {
//        std::cout << "Error validating test failed!" << std::endl;
//      }
//      else
//      {
//        std::cout << "Test PASSED" << std::endl;
//      }
//
//    }

    std::cout << (runSequential ? "sequential" : "htgs") << ", " << (runSequential ? numBlasThreads : numProdThreads)
              << ", width-b: " << matrixBWidth << ", height-a: " << matrixAHeight
              << ", shared-dim: " << sharedDim
              << ", " << ", blockSize: " << (runSequential ? 0 : blockSize) << ", time:"
              << clk.getAverageTime(TimeVal::MILLI)
              << std::endl;

    runtimeFile << (runSequential ? "sequential" : "htgs") << ", " << (runSequential ? numBlasThreads : numProdThreads)
                << ", "
                << matrixBWidth << ", " << matrixAHeight
                << ", " << sharedDim << ", " << blockSize << ", " << clk.getAverageTime(TimeVal::MILLI)
                << std::endl;

  }

  delete[] matrixA;
  delete[] matrixB;
  delete[] matrixC;

}