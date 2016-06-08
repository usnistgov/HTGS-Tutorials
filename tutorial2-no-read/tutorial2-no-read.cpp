//
// Created by tjb3 on 2/23/16.
//

#include <htgs/api/TaskGraph.hpp>
#include <htgs/api/Runtime.hpp>
#include "data/MatrixRequestData.h"
#include "data/MatrixBlockData.h"
#include "tasks/GenMatrixTask.h"
#include "rules/MatrixLoadRule.h"
#include "tasks/HadamardProductTask.h"
#include "../tutorial-utils/SimpleClock.h"

int main(int argc, char *argv[])
{

  int width = 4096;
  int height = 4096;
  int blockSize = 256;
  int numReadThreads = 1;
  int numProdThreads = 1;

  int numRetry = 5;

  for (int arg = 1; arg < argc; arg++)
  {
    std::string argvs(argv[arg]);

    if (argvs == "--width")
    {
      arg++;
      width = atoi(argv[arg]);
    }

    if (argvs == "--height")
    {
      arg++;
      height = atoi(argv[arg]);
    }

    if (argvs == "--block-size")
    {
      arg++;
      blockSize = atoi(argv[arg]);
    }

    if (argvs == "--num-readers")
    {
      arg++;
      numReadThreads = atoi(argv[arg]);
    }

    if (argvs == "--num-workers")
    {
      arg++;
      numProdThreads = atoi(argv[arg]);
    }

    if (argvs == "--help")
    {
      std::cout << argv[0] << " help: [--width <#>] [--height <#>] [--block-size <#>] [--num-readers <#>] [--num-workers <#>] [--help]" << std::endl;
      exit(0);
    }
  }
  SimpleClock clk;

  for (int i =0; i < numRetry; i++) {
    GenMatrixTask *genMatTask = new GenMatrixTask(numReadThreads, blockSize, width, height);
    MatrixMulBlkTask *prodTask = new MatrixMulBlkTask(numProdThreads);

    int numBlocksCols = genMatTask->getNumBlocksCols();
    int numBlocksRows = genMatTask->getNumBlocksRows();

    MatrixLoadRule *loadRule = new MatrixLoadRule(numBlocksCols, numBlocksRows);
    auto bookkeeper = new htgs::Bookkeeper<MatrixBlockData<double *>>();

    auto taskGraph = new htgs::TaskGraph<MatrixRequestData, MatrixBlockData<double *>>();

    taskGraph->addGraphInputConsumer(genMatTask);
    taskGraph->addEdge(genMatTask, bookkeeper);
    taskGraph->addRule(bookkeeper, prodTask, loadRule);
    taskGraph->addGraphOutputProducer(prodTask);

    taskGraph->incrementGraphInputProducer();

    htgs::Runtime *runtime = new htgs::Runtime(taskGraph);

    clk.start();

    runtime->executeRuntime();

    for (int row = 0; row < numBlocksRows; row++) {
      for (int col = 0; col < numBlocksCols; col++) {
        MatrixRequestData *matrixA = new MatrixRequestData(row, col, MatrixType::MatrixA);
        MatrixRequestData *matrixB = new MatrixRequestData(row, col, MatrixType::MatrixB);

        taskGraph->produceData(matrixA);
        taskGraph->produceData(matrixB);
      }
    }

    taskGraph->finishedProducingData();

    while (!taskGraph->isOutputTerminated()) {
      auto data = taskGraph->consumeData();

      if (data != nullptr) {
        double *mem = data->getMatrixData();
        delete[] mem;
      }

    }

    taskGraph->finishReleasingMemory();

    runtime->waitForRuntime();

    clk.stopAndIncrement();
    delete runtime;
  }
  std::cout << "width: " << width << ", height: " << height << ", blocksize: " << blockSize << ", average time: " << clk.getAverageTime(TimeVal::MILLI) << " ms" << std::endl;

  std::cout.flush();



}