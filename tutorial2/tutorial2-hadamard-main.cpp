
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 2/23/16.
//

#include <htgs/api/TaskGraphConf.hpp>
#include <htgs/api/TaskGraphRuntime.hpp>
#include "../tutorial-utils/matrix-library/data/MatrixRequestData.h"
#include "../tutorial-utils/matrix-library/data/MatrixBlockData.h"
#include "../tutorial-utils/matrix-library/tasks/ReadDiskMatrixTask.h"
#include "../tutorial-utils/matrix-library/allocator/MatrixAllocator.h"
#include "rules/HadamardLoadRule.h"
#include "tasks/HadamardProductTask.h"
#include "../tutorial-utils/SimpleClock.h"
#include "../tutorial-utils/util-matrix.h"

int main(int argc, char *argv[]) {
  size_t width = 1024;
  size_t height = 1024;
  size_t blockSize = 256;
  size_t numReadThreads = 1;
  size_t numProdThreads = 10;
  std::string directory("data");

  for (int arg = 1; arg < argc; arg++) {
    std::string argvs(argv[arg]);

    if (argvs == "--width") {
      arg++;
      width = (size_t)atoi(argv[arg]);
    }

    if (argvs == "--height") {
      arg++;
      height = (size_t)atoi(argv[arg]);
    }

    if (argvs == "--block-size") {
      arg++;
      blockSize = (size_t)atoi(argv[arg]);
    }

    if (argvs == "--num-readers") {
      arg++;
      numReadThreads = (size_t)atoi(argv[arg]);
    }

    if (argvs == "--num-workers") {
      arg++;
      numProdThreads = (size_t)atoi(argv[arg]);
    }

    if (argvs == "--dir") {
      arg++;
      directory = argv[arg];
    }

    if (argvs == "--help") {
      std::cout << argv[0]
                << " help: [--width <#>] [--height <#>] [--block-size <#>] [--num-readers <#>] [--num-workers <#>] [--dir <dir>] [--help]"
                << std::endl;
      exit(0);
    }
  }

  // Check directory for matrix files based on the given file size
  checkAndValidateMatrixBlockFiles(directory, width, height, width, height, blockSize, false);

  ReadDiskMatrixTask *readMatTask = new ReadDiskMatrixTask(numReadThreads, blockSize, width, height, directory);
  HadamardProductTaskWithMemEdge *prodTask = new HadamardProductTaskWithMemEdge(numProdThreads);

  size_t numBlocksCols = readMatTask->getNumBlocksCols();
  size_t numBlocksRows = readMatTask->getNumBlocksRows();

  std::shared_ptr<HadamardLoadRule<htgs::m_data_t<double>>> loadRule = std::make_shared<HadamardLoadRule<htgs::m_data_t<double>>>(numBlocksCols, numBlocksRows);
  auto bookkeeper = new htgs::Bookkeeper<MatrixBlockData<htgs::m_data_t<double>>>();

  auto taskGraph = new htgs::TaskGraphConf<MatrixRequestData, MatrixBlockData<htgs::m_data_t<double>>>();

  taskGraph->setGraphConsumerTask(readMatTask);
  taskGraph->addEdge(readMatTask, bookkeeper);
  taskGraph->addRuleEdge(bookkeeper, loadRule, prodTask);
  taskGraph->addGraphProducerTask(prodTask);

  std::shared_ptr<MatrixAllocator<double>> matAlloc = std::make_shared<MatrixAllocator<double>>(blockSize, blockSize);

  taskGraph->addMemoryManagerEdge<double>("result", prodTask, matAlloc, 50, htgs::MMType::Static);

  taskGraph->addMemoryManagerEdge<double>("matrixA", readMatTask, matAlloc, 100, htgs::MMType::Static);
  taskGraph->addMemoryManagerEdge<double>("matrixB", readMatTask, matAlloc, 100, htgs::MMType::Static);


  htgs::TaskGraphRuntime *runtime = new htgs::TaskGraphRuntime(taskGraph);

  SimpleClock clk;
  clk.start();

  runtime->executeRuntime();

  for (size_t row = 0; row < numBlocksRows; row++) {
    for (size_t col = 0; col < numBlocksCols; col++) {
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
      std::cout << "Result received: " << data->getRequest()->getRow() << ", " << data->getRequest()->getCol()
                << std::endl;

      taskGraph->releaseMemory(data->getMatrixData());
    }
  }

  runtime->waitForRuntime();

  clk.stopAndIncrement();

  std::cout << "width: " << width << ", height: " << height << ", blocksize: " << blockSize << ", time: "
            << clk.getAverageTime(TimeVal::MILLI) << " ms" << std::endl;

  delete runtime;
}