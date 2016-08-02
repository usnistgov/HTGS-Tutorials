
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 2/23/16.
//
//#define DEBUG_FLAG
//#define DEBUG_LEVEL_VERBOSE
//#define PROFILE
#include <htgs/api/TaskGraph.hpp>
#include <htgs/api/Runtime.hpp>
#include <cblas.h>
#include <iomanip>
#include <cfloat>

#include "data/MatrixRequestData.h"
#include "data/MatrixBlockData.h"
#include "rules/GausElimRuleUpper.h"
#include "rules/GausElimRuleLower.h"
#include "rules/UpdateRule.h"
#include "rules/GausElimRule.h"

#include "../tutorial-utils/SimpleClock.h"
#include "../tutorial-utils/util-matrix.h"
#include "tasks/GausElimTask.h"
#include "tasks/FactorUpperTask.h"
#include "tasks/FactorLowerTask.h"
#include "rules/MatrixMulRule.h"
#include "tasks/MatrixMulBlkTask.h"

int validateResults(double *luMatrix, double *origMatrix, int matrixSize) {
  int count = 0;

  // Multiply lower triangle with upper triangle
//  double *result = new double[matrixSize *matrixSize];

//  std::cout<< "orig matrix:" << std::endl;

  double *lMatrix = new double[matrixSize * matrixSize];
  double *uMatrix = new double[matrixSize * matrixSize];

  for (int c = 0; c < matrixSize; c++)
  {
    for (int r = 0; r < matrixSize; r++)
    {
      // below diag
      if (r > c)
      {
        lMatrix[IDX2C(r, c, matrixSize)] = luMatrix[IDX2C(r, c, matrixSize)];
        uMatrix[IDX2C(r, c, matrixSize)] = 0.0;
      }
        // above diag
      else if (c > r)
      {
        lMatrix[IDX2C(r, c, matrixSize)] = 0.0;
        uMatrix[IDX2C(r, c, matrixSize)] = luMatrix[IDX2C(r, c, matrixSize)];
      }
        // on diag
      else if (r == c)
      {
        lMatrix[IDX2C(r, c, matrixSize)] = 1.0;
        uMatrix[IDX2C(r, c, matrixSize)] = luMatrix[IDX2C(r, c, matrixSize)];
      }
    }
  }

//  std::cout << "orig matrix:" << std::endl;
//  for (int r = 0; r < matrixSize; r++)
//  {
//    for (int c = 0; c < matrixSize; c++)
//    {
//      std::cout << std::setw(14) << std::setprecision(9) << origMatrix[IDX2C(r, c, matrixSize)] << " ";
//    }
//    std::cout <<std::endl;
//  }
//  std::cout << "U matrix:" << std::endl;
//  for (int r = 0; r < matrixSize; r++)
//  {
//    for (int c = 0; c < matrixSize; c++)
//    {
//      std::cout << std::setw(14) << std::setprecision(9) << uMatrix[IDX2C(r, c, matrixSize)] << " ";
//    }
//    std::cout <<std::endl;
//  }


//  std::cout << "L matrix:" << std::endl;
//  for (int r = 0; r < matrixSize; r++)
//  {
//    for (int c = 0; c < matrixSize; c++)
//    {
//      std::cout << std::setw(14) << std::setprecision(9) << lMatrix[IDX2C(r, c, matrixSize)] << " ";
//    }
//    std::cout <<std::endl;
//  }

  // Create identiy matrix
  double *result = new double[matrixSize*matrixSize];

  openblas_set_num_threads(40);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, matrixSize, matrixSize, matrixSize, 1.0, lMatrix, matrixSize, uMatrix, matrixSize, 0.0, result, matrixSize);

//  std::cout << "result matrix:" << std::endl;
//  for (int r = 0; r < matrixSize; r++)
//  {
//    for (int c = 0; c < matrixSize; c++)
//    {
//      std::cout << std::setw(14) << std::setprecision(9) << result[IDX2C(r, c, matrixSize)] << " ";
//    }
//    std::cout <<std::endl;
//  }

  for (int r = 0; r < matrixSize; r++)
  {
    for (int c = 0; c < matrixSize; c++)
    {

      double difference = abs(result[IDX2C(r, c, matrixSize)] - origMatrix[IDX2C(r, c, matrixSize)]);
      if (difference > DBL_EPSILON)
      {
        count++;
        if (count < 20)
        {
          std::cout << "Incorrect value: " << result[IDX2C(r, c, matrixSize)] << " != " << origMatrix[IDX2C(r, c, matrixSize)] << std::endl;
        }
      }
    }
  }

  if (count > 0)
    std::cout << "Total incorrect = " << count << std::endl;

  if (count > 0)
    return 1;
  return 0;
}

int main(int argc, char *argv[]) {
  long matrixSize= 16384;
  int blockSize = 128;
  bool runSequential = false;
  bool validate = false;

  int numGausElimThreads = 4;
  int numFactorLowerThreads = 8;
  int numFactorUpperThreads = 8;
  int numMatrixMulThreads = 20;

  std::string runtimeFileStr("runtimes");

  int numRetry = 1;

  if (argc > 1) {
    for (int arg = 1; arg < argc; arg++) {
      std::string argvs(argv[arg]);

      if (argvs == "--size") {
        arg++;
        matrixSize = atoi(argv[arg]);
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


      if (argvs == "--runtime-file" && arg + 1 < argc) {
        runtimeFileStr = argv[arg + 1];
        arg++;
      }

//      if (argvs == "--validate-results") {
//        validate = true;
//      }

      if (argvs == "--help") {
        std::cout << argv[0]
                  << " args: [--size <#>] [--block-size <#>] [--num-retry <#>] [--runtime-file <filename>] [--validate-results] [--run-sequential] [--help]"
                  << std::endl;
        exit(0);
      }
    }
  }

  std::ofstream runtimeFile(runtimeFileStr, std::ios::app);
  double *matrix = new double[matrixSize * matrixSize];
  double *matrixTest = new double[matrixSize * matrixSize];

  // TODO: Ensure diagonally dominant
  initMatrixDiagDom(matrix, matrixSize, matrixSize, true);

  for (int i = 0; i < matrixSize*matrixSize; i++)
    matrixTest[i] = matrix[i];

  for (int numTry = 0; numTry < numRetry; numTry++) {
    SimpleClock clk;
    SimpleClock endToEnd;

    if (runSequential) {
      endToEnd.start();
      openblas_set_num_threads(40);

//      clk.start();
//      computeSequentialMatMul(matrixA, matrixB, matrixC, matrixAHeight, sharedDim, matrixBWidth);
//      clk.stopAndIncrement();
      endToEnd.stopAndIncrement();
    }
    else {
      endToEnd.start();
      openblas_set_num_threads(1);

      int gridHeight = (int) matrixSize / blockSize;
      int gridWidth = (int) matrixSize / blockSize;

      // TODO: Build graph and runtime
      htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>> *matrixBlocks = new htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>>(gridHeight, gridWidth, nullptr);

      for (int r = 0; r < gridHeight; r++)
      {
        for (int c = 0; c < gridWidth; c++)
        {
          // Store pointer locations for all blocks
          double *ptr = &matrix[IDX2C(r * blockSize, c *blockSize, matrixSize)];

          std::shared_ptr<MatrixRequestData> request(new MatrixRequestData(r, c, MatrixType::MatrixA));
          std::shared_ptr<MatrixBlockData<double *>> data(new MatrixBlockData<double *>(request, ptr, blockSize, blockSize));

          matrixBlocks->set(r, c, data);
        }
      }

      GausElimTask *gausElimTask = new GausElimTask(numGausElimThreads, matrixSize, matrixSize);

      auto gausElimBk = new htgs::Bookkeeper<MatrixBlockData<double *>>();

      GausElimRuleUpper *gausElimRuleUpper = new GausElimRuleUpper(matrixBlocks, gridHeight, gridWidth);
      GausElimRuleLower *gausElimRuleLower = new GausElimRuleLower(matrixBlocks, gridHeight, gridWidth);

      FactorUpperTask *factorUpperTask = new FactorUpperTask(numFactorUpperThreads, matrixSize, matrixSize);
      FactorLowerTask *factorLowerTask = new FactorLowerTask(numFactorLowerThreads, matrixSize, matrixSize);

      auto matrixMulBk = new htgs::Bookkeeper<MatrixBlockData<double *>>();
      MatrixMulRule *matrixMulRule = new MatrixMulRule(matrixBlocks, gridHeight, gridWidth);

      MatrixMulBlkTask *matrixMulTask = new MatrixMulBlkTask(numMatrixMulThreads, matrixSize, matrixSize, matrixSize, matrixSize, blockSize);


      auto matrixMulResultBk = new htgs::Bookkeeper<MatrixBlockData<double *>>();

      int numDiagonals = gridWidth - 1;
      GausElimRule *gausElimRule = new GausElimRule(numDiagonals, gridHeight, gridWidth);

      // Number of updates excluding the diagonal and the top/right row/column
      int numUpdates = (1.0/6.0) * (double)gridWidth * (2.0 * ((double)gridWidth * (double)gridWidth) - 3.0 * (double)gridWidth + 1.0);
//      if (gridWidth * gridHeight == 1)
//        numUpdates = 0;
//      else
//        numUpdates= gridWidth * gridHeight - (gridWidth - 1) - (gridHeight - 1) - gridWidth;

      UpdateRule *updateRule = new UpdateRule(numUpdates);
      UpdateRule *updateRule2 = new UpdateRule(numUpdates);

      auto taskGraph = new htgs::TaskGraph<MatrixBlockData<double *>, htgs::VoidData>();
      taskGraph->addGraphInputConsumer(gausElimTask);

      taskGraph->addEdge(gausElimTask, gausElimBk);
      taskGraph->addRule(gausElimBk, factorUpperTask, gausElimRuleUpper);
      taskGraph->addRule(gausElimBk, factorLowerTask, gausElimRuleLower);

      taskGraph->addEdge(factorUpperTask, matrixMulBk);
      taskGraph->addEdge(factorLowerTask, matrixMulBk);

      taskGraph->addRule(matrixMulBk, matrixMulTask, matrixMulRule);
      taskGraph->addEdge(matrixMulTask, matrixMulResultBk);

      if (numDiagonals > 0)
        taskGraph->addRule(matrixMulResultBk, gausElimTask, gausElimRule);

      if (numUpdates > 0)
        taskGraph->addRule(matrixMulResultBk, matrixMulBk, updateRule);

      if (numUpdates > 0)
        taskGraph->addRule(matrixMulResultBk, gausElimBk, updateRule2);

      taskGraph->incrementGraphInputProducer();

      taskGraph->writeDotToFile("lud-graph.dot");

      htgs::Runtime *runtime = new htgs::Runtime(taskGraph);

      clk.start();

      runtime->executeRuntime();

      taskGraph->produceData(matrixBlocks->get(0, 0));
      taskGraph->finishedProducingData();

//      for (int i = 0; i < 9000; i++)
//        std::cout << i << std::endl;
//
//      taskGraph->writeDotToFile("lud-graph-exec.dot");

      runtime->waitForRuntime();

      // TODO: Run

      clk.stopAndIncrement();


      delete runtime;
      endToEnd.stopAndIncrement();
    }

    // TODO: Validate
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

    double operations = (2.0 * (matrixSize * matrixSize * matrixSize)) / 3.0;
    double flops = operations / clk.getAverageTime(TimeVal::SEC);
    double gflops = flops / 1073741824.0;




    std::cout << (runSequential ? "sequential" : "htgs") << ", matrix-size: " << matrixSize
              << ", " << "blockSize: " << (runSequential ? 0 : blockSize) << ", time:"
              << clk.getAverageTime(TimeVal::MILLI)
              << ", end-to-end:" << endToEnd.getAverageTime(TimeVal::MILLI) << " gflops: " << gflops

        << std::endl;

    runtimeFile << (runSequential ? "sequential" : "htgs")
                << ", " << matrixSize
                << ", " << blockSize << ", " << clk.getAverageTime(TimeVal::MILLI)
                << ", " << endToEnd.getAverageTime(TimeVal::MILLI)
                << std::endl;



    if (validate)
    {
      int res = validateResults(matrix, matrixTest, matrixSize);
      std::cout << (res == 0 ? "PASSED" : "FAILED") << std::endl;
    }


  }

  delete[] matrix;
  delete[] matrixTest;

}