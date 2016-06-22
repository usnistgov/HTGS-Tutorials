//
// Created by tjb3 on 2/23/16.
//

#include <htgs/api/TaskGraph.hpp>
#include <htgs/api/Runtime.hpp>
#include "data/MatrixRequestData.h"
#include "data/MatrixBlockData.h"
#include "tasks/ReadMatrixTask.h"
#include "memory/MatrixAllocator.h"
#include "rules/MatrixLoadRule.h"
#include "tasks/HadamardProductTask.h"
#include "../../tutorial-utils/SimpleClock.h"
#include "../../tutorial-utils/util-matrix.h"

int main(int argc, char *argv[])
{
  int width = 1024;
  int height = 1024;
  int blockSize = 256;
  int numReadThreads = 1;
  int numProdThreads = 10;
  std::string directory("data");

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

    if (argvs == "--dir")
    {
      arg++;
      directory = argv[arg];
    }

    if (argvs == "--help")
    {
      std::cout << argv[0] << " help: [--width <#>] [--height <#>] [--block-size <#>] [--num-readers <#>] [--num-workers <#>] [--dir <dir>] [--help]" << std::endl;
      exit(0);
    }
  }



  // Check directory for matrix files based on the given file size
  checkAndValidateMatrixBlockFiles(directory, width, height, width, height, blockSize, false);


  ReadMatrixTask *readMatTask = new ReadMatrixTask(numReadThreads, blockSize, width, height, directory);
  MatrixMulBlkTask *prodTask = new MatrixMulBlkTask(numProdThreads);

  int numBlocksCols = readMatTask->getNumBlocksCols();
  int numBlocksRows = readMatTask->getNumBlocksRows();

  MatrixLoadRule *loadRule = new MatrixLoadRule(numBlocksCols, numBlocksRows);
  auto bookkeeper = new htgs::Bookkeeper<MatrixBlockData<MatrixMemoryData_t>>();

  auto taskGraph = new htgs::TaskGraph<MatrixRequestData, MatrixBlockData<double *>>();

  taskGraph->addGraphInputConsumer(readMatTask);
  taskGraph->addEdge(readMatTask, bookkeeper);
  taskGraph->addRule(bookkeeper, prodTask, loadRule);
  taskGraph->addGraphOutputProducer(prodTask);

  taskGraph->addGraphUserManagedMemoryManagerEdge("outputMem", prodTask, 50);

  taskGraph->addMemoryManagerEdge("matrixA", readMatTask, prodTask, new MatrixAllocator(blockSize, blockSize), 100, htgs::MMType::Static);
  taskGraph->addMemoryManagerEdge("matrixB", readMatTask, prodTask, new MatrixAllocator(blockSize, blockSize), 100, htgs::MMType::Static);

  taskGraph->incrementGraphInputProducer();

  htgs::Runtime *runtime = new htgs::Runtime(taskGraph);

  SimpleClock clk;
  clk.start();

  runtime->executeRuntime();

  for (int row = 0; row < numBlocksRows; row++)
  {
    for (int col = 0; col < numBlocksCols; col++)
    {
      MatrixRequestData * matrixA = new MatrixRequestData(row, col, MatrixType::MatrixA);
      MatrixRequestData * matrixB = new MatrixRequestData(row, col, MatrixType::MatrixB);

      taskGraph->produceData(matrixA);
      taskGraph->produceData(matrixB);
    }
  }

  taskGraph->finishedProducingData();

  while (!taskGraph->isOutputTerminated())
  {
    auto data = taskGraph->consumeData();

    if (data != nullptr) {
      std::cout << "Result received: " << data->getRequest()->getRow() << ", " << data->getRequest()->getCol() <<std::endl;
      double *mem = data->getMatrixData();
      delete [] mem;

      taskGraph->memRelease("outputMem", 0);
    }
  }

  taskGraph->finishReleasingMemory();

  runtime->waitForRuntime();

  clk.stopAndIncrement();

  std::cout << "width: " << width << ", height: " << height << ", blocksize: " << blockSize << ", time: " << clk.getAverageTime(TimeVal::MILLI) << " ms" << std::endl;

  delete runtime;
}