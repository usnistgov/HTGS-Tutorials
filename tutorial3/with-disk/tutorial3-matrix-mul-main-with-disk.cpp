
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
#include <string.h>

#include "../../tutorial-utils/matrix-library/data/MatrixRequestData.h"
#include "../../tutorial-utils/matrix-library/data/MatrixBlockData.h"
#include "../../tutorial-utils/matrix-library/tasks/ReadDiskMatrixTask.h"
#include "../../tutorial-utils/matrix-library/allocator/MatrixAllocator.h"
#include "../rules/MatMulLoadRule.h"
#include "tasks/MatMulBlkTaskWithMemRelease.h"
#include "../tasks/MatMulAccumTask.h"
#include "tasks/MatMulOutputTaskWithDisk.h"
#include "../rules/MatMulAccumulateRule.h"
#include "../rules/MatMulDistributeRule.h"
#include "../../tutorial-utils/SimpleClock.h"
#include "../../tutorial-utils/util-matrix.h"
#include "../../tutorial-utils/matrix-library/args/MatMulArgs.h"

int validateResults(std::string baseDirectory, size_t matrixAHeight, size_t matrixBWidth, size_t blockSize) {
  size_t blkHeightMatA = (size_t) ceil((double) matrixAHeight / (double) blockSize);
  size_t blkWidthMatB = (size_t) ceil((double) matrixBWidth / (double) blockSize);
  double *cMem = new double[matrixAHeight * matrixBWidth];
  double *cMemPar = new double[matrixAHeight * matrixBWidth];

  for (size_t row = 0; row < blkHeightMatA; row++) {
    for (size_t col = 0; col < blkWidthMatB; col++) {
      matrixAHeight = (row == blkHeightMatA - 1 && matrixAHeight % blockSize != 0) ? matrixAHeight % blockSize : blockSize;
      matrixBWidth = (col == blkWidthMatB - 1 && matrixBWidth % blockSize != 0) ? matrixBWidth % blockSize : blockSize;

      std::string fileName(baseDirectory + "/matrixC/" + std::to_string(row) + "_" + std::to_string(col));
      std::string fileNamePar(baseDirectory + "/matrixC_HTGS/" + std::to_string(row) + "_" + std::to_string(col));

      std::ifstream c(fileName, std::ios::binary);
      std::ifstream cPar(fileNamePar, std::ios::binary);

      c.read((char *) cMem, sizeof(double) * matrixAHeight * matrixBWidth);
      cPar.read((char *) cMemPar, sizeof(double) * matrixAHeight * matrixBWidth);

      if (!validateMatMulResults(20, cMem, cMemPar, matrixAHeight * matrixBWidth))
      {
        delete []cMem;
        delete []cMemPar;
        return -1;
      }
    }
  }

  delete []cMem;
  delete []cMemPar;

  return 0;
}

void computeSequentialMatMul(std::string directoryA,
                             std::string directoryB,
                             std::string outputDirectory,
                             size_t fullMatrixAHeight,
                             size_t fullMatrixAWidth,
                             size_t fullMatrixBWidth,
                             size_t blockSize) {
  std::string matrixCDir(outputDirectory + "/matrixC");
  create_dir(matrixCDir);
  size_t blkHeightMatA = (size_t) ceil((double) fullMatrixAHeight / (double) blockSize);
  size_t blkWidthMatA = (size_t) ceil((double) fullMatrixAWidth / (double) blockSize);
  size_t blkHeightMatB = blkWidthMatA;
  size_t blkWidthMatB = (size_t) ceil((double) fullMatrixBWidth / (double) blockSize);

  double ***matALookup = new double **[blkHeightMatA];
  for (size_t i = 0; i < blkHeightMatA; i++) {
    matALookup[i] = new double *[blkWidthMatA];
  }

  double ***matBLookup = new double **[blkHeightMatB];
  for (size_t i = 0; i < blkHeightMatB; i++) {
    matBLookup[i] = new double *[blkWidthMatB];
  }

  for (size_t i = 0; i < blkHeightMatA; i++) {
    for (size_t j = 0; j < blkWidthMatA; j++) {
      matALookup[i][j] = nullptr;
    }
  }

  for (size_t i = 0; i < blkHeightMatB; i++) {
    for (size_t j = 0; j < blkWidthMatB; j++) {
      matBLookup[i][j] = nullptr;
    }
  }

  for (size_t blkRowA = 0; blkRowA < blkHeightMatA; blkRowA++) {
    for (size_t blkColB = 0; blkColB < blkWidthMatB; blkColB++) {
      size_t matrixAHeight =
          (blkRowA == blkHeightMatA - 1 && fullMatrixAHeight % blockSize != 0) ? fullMatrixAHeight % blockSize
                                                                               : blockSize;
      size_t matrixBWidth =
          (blkColB == blkWidthMatB - 1 && fullMatrixBWidth % blockSize != 0) ? fullMatrixBWidth % blockSize : blockSize;

      std::string matrixCFile(matrixCDir + "/" + std::to_string(blkRowA) + "_" + std::to_string(blkColB));

      double *finalResultC = new double[matrixAHeight * matrixBWidth];
      memset(finalResultC, 0, sizeof(double) * matrixAHeight * matrixBWidth);
      // matrix C . . .
      for (size_t blk = 0; blk < blkWidthMatA; blk++) {
        // Read A and B
        size_t matrixAWidth =
            (blk == blkWidthMatA - 1 && fullMatrixAWidth % blockSize != 0) ? fullMatrixAWidth % blockSize : blockSize;

        double *matrixA;
        double *matrixB;
        if (matALookup[blkRowA][blk] == nullptr) {
          matrixA = new double[matrixAHeight * matrixAWidth];
          std::string matrixAFile(directoryA + "/MatrixA/" + std::to_string(blkRowA) + "_" + std::to_string(blk));
          std::ifstream fileA(matrixAFile, std::ios::binary);
          fileA.read((char *) matrixA, sizeof(double) * matrixAHeight * matrixAWidth);
          matALookup[blkRowA][blk] = matrixA;
        }
        else {
          matrixA = matALookup[blkRowA][blk];
        }

        if (matBLookup[blk][blkColB] == nullptr) {
          matrixB = new double[matrixBWidth * matrixAWidth];
          std::string matrixBFile(directoryB + "/MatrixB/" + std::to_string(blk) + "_" + std::to_string(blkColB));
          std::ifstream fileB(matrixBFile, std::ios::binary);
          fileB.read((char *) matrixB, sizeof(double) * matrixBWidth * matrixAWidth);
          matBLookup[blk][blkColB] = matrixB;
        }
        else {
          matrixB = matBLookup[blk][blkColB];
        }

        std::cout << "Seq Computing A(" << blkRowA << ", " << blk << ") x B(" << blk << ", " << blkColB << ") = C("
                  << blkRowA << ", " << blkColB << ")" << std::endl;

        for (size_t i = 0; i < matrixAHeight; i++) {
          for (size_t j = 0; j < matrixBWidth; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < matrixAWidth; k++) {
              sum += matrixA[i * matrixAWidth + k] * matrixB[k * matrixBWidth + j];

            }
            finalResultC[i * matrixBWidth + j] += sum;
          }

        }
      }

      std::ofstream fileC(matrixCFile, std::ios::binary);
      fileC.write((char *) finalResultC, sizeof(double) * matrixAHeight * matrixBWidth);

      delete[] finalResultC;
    }
  }

  for (size_t i = 0; i < blkHeightMatA; i++) {
    for (size_t j = 0; j < blkWidthMatA; j++) {
      delete[] matALookup[i][j];
    }
    delete[] matALookup[i];
  }

  delete[]matALookup;

  for (size_t i = 0; i < blkHeightMatB; i++) {
    for (size_t j = 0; j < blkWidthMatB; j++) {
      delete[] matBLookup[i][j];
    }
    delete[] matBLookup[i];
  }

  delete[]matBLookup;
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
  size_t numAccumThreads = (size_t)ceil((double)numProdThreads / 2.0);
  std::string directory = matMulArgs.getDirectory();
  std::string outputDirectory = matMulArgs.getOutputDir();
  bool runSequential = matMulArgs.isRunSequential();
  bool validate = matMulArgs.isValidateResults();

  create_dir(outputDirectory);

  checkAndValidateMatrixBlockFiles(directory, sharedDim, matrixAHeight, matrixBWidth, sharedDim, blockSize, false);

  std::string inputDirectoryA = generateDirectoryName(directory, sharedDim, matrixAHeight, blockSize);
  std::string inputDirectoryB = generateDirectoryName(directory, matrixBWidth, sharedDim, blockSize);

  outputDirectory = generateDirectoryName(outputDirectory, matrixBWidth, matrixAHeight, blockSize);

  SimpleClock clk;

  if (runSequential) {
    clk.start();
    computeSequentialMatMul(inputDirectoryA,
                            inputDirectoryB,
                            outputDirectory,
                            matMulArgs.getMatrixAHeight(),
                            matMulArgs.getSharedDim(),
                            matMulArgs.getMatrixBWidth(),
                            matMulArgs.getBlockSize());
  }
  else {

    ReadDiskMatrixTask *readAMatTask = new ReadDiskMatrixTask(numReadThreads, blockSize, sharedDim, matrixAHeight, inputDirectoryA, MatrixType::MatrixA, true);
    ReadDiskMatrixTask *readBMatTask = new ReadDiskMatrixTask(numReadThreads, blockSize, matrixBWidth, sharedDim, inputDirectoryB, MatrixType::MatrixB, true);

    MatMulBlkTaskWithMemRelease *mmulTask = new MatMulBlkTaskWithMemRelease(numProdThreads, false);
    MatMulAccumTask *accumTask = new MatMulAccumTask(numAccumThreads, false);

    MatMulOutputTaskWithDisk *outputTask = new MatMulOutputTaskWithDisk(outputDirectory);

    size_t blkHeightMatB = readBMatTask->getNumBlocksRows();
    size_t blkWidthMatB = readBMatTask->getNumBlocksCols();

    size_t blkHeightMatA = readAMatTask->getNumBlocksRows();
    size_t blkWidthMatA = readAMatTask->getNumBlocksCols();

    std::cout << "matA: " << blkHeightMatA << ", " << blkWidthMatA << " :: matB: " << blkHeightMatB << ", "
              << blkWidthMatB << std::endl;

    MatMulDistributeRule *distributeRuleMatA = new MatMulDistributeRule(MatrixType::MatrixA);
    MatMulDistributeRule *distributeRuleMatB = new MatMulDistributeRule(MatrixType::MatrixB);

    MatMulLoadRule<htgs::m_data_t<double>> *loadRule = new MatMulLoadRule<htgs::m_data_t<double>>(blkWidthMatA, blkHeightMatA, blkWidthMatB, blkHeightMatB);
    MatMulAccumulateRule<double *> *accumulateRule = new MatMulAccumulateRule<double *>(blkWidthMatB, blkHeightMatA, blkWidthMatA);

    MatMulOutputRule *outputRule = new MatMulOutputRule(blkWidthMatB, blkHeightMatA, blkWidthMatA);

    auto distributeBk = new htgs::Bookkeeper<MatrixRequestData>();
    auto matMulBk = new htgs::Bookkeeper<MatrixBlockData<htgs::m_data_t<double>>>();
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


    MatrixAllocator<double> *matAlloc = new MatrixAllocator<double>(blockSize, blockSize);

    taskGraph->addMemoryManagerEdge("MatrixA",
                                    readAMatTask,
                                    matAlloc,
                                    1000,
                                    htgs::MMType::Static);
    taskGraph->addMemoryManagerEdge("MatrixB",
                                    readBMatTask,
                                    matAlloc,
                                    1000,
                                    htgs::MMType::Static);


    htgs::TaskGraphRuntime *runtime = new htgs::TaskGraphRuntime(taskGraph);

    clk.start();

    runtime->executeRuntime();

    for (size_t row = 0; row < blkHeightMatA; row++) {
      for (size_t col = 0; col < blkWidthMatA; col++) {

        MatrixRequestData *matrixA = new MatrixRequestData(row, col, MatrixType::MatrixA);
        taskGraph->produceData(matrixA);

      }
    }

    for (size_t col = 0; col < blkWidthMatB; col++) {
      for (size_t row = 0; row < blkHeightMatB; row++) {

        MatrixRequestData *matrixB = new MatrixRequestData(row, col, MatrixType::MatrixB);
        taskGraph->produceData(matrixB);

      }
    }

    taskGraph->finishedProducingData();


    while (!taskGraph->isOutputTerminated()) {
      auto data = taskGraph->consumeData();
      if (data != nullptr) {
//      std::cout << "Result received: " << data->getRow() << ", " << data->getCol() <<std::endl;
//        taskGraph->writeDotToFile("matmul.dot");

      }
    }

    runtime->waitForRuntime();

    delete runtime;
  }

  clk.stopAndIncrement();

  if (validate) {
    int res = validateResults(directory, matrixAHeight, matrixBWidth, blockSize);
    std::cout << "Finished (" << (res != 0 ? "FAILED - must run sequential" : "PASSED") << ") ";
  }

  std::cout << (runSequential ? "Sequential, " : "Parallel, ")
            << "width-b: " << matrixBWidth << ", height-a: " << matrixAHeight
            << ", shared-dim: " << sharedDim << ", blocksize: " << blockSize << ", time: " <<
            clk.getAverageTime(TimeVal::MILLI)
            << std::endl;

}