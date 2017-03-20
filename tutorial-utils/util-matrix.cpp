
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 1/15/16.
//

#include <random>
#include <string.h>

#include "util-matrix.h"
#include "util-filesystem.h"

double *allocMatrix(size_t width, size_t height) {
  double *matrix = new double[height * width];
  return matrix;
}

void initMatrixDiagDom(double * matrix, long width, long height, bool columnStore)
{
  long min = 1;
  long max = 42;
  unsigned long seed = 9000;


  long minDiag = max * width + 10;
  long maxDiag = minDiag + 9000;

  std::uniform_int_distribution<long> unif(min, max);
  std::default_random_engine re(seed);

  std::uniform_int_distribution<long> unifDiag(minDiag, maxDiag);

  if (columnStore) {
    for (long c = 0; c < width; c++) {
      for (long r = 0; r < height; r++) {
        if (r == c)
          matrix[IDX2C(r, c, height)] = unifDiag(re);
        else
          matrix[IDX2C(r, c, height)] = unif(re);
      }
    }
  }
  else {
    for (long r = 0; r < height; r++) {
      for (long c = 0; c < width; c++) {
        if (r == c)
          matrix[r * width + c] = unifDiag(re);
        else
          matrix[r * width + c] = unif(re);
      }
    }
  }

}

void initMatrix(double *matrix, long width, long height, bool columnStore) {
  long min = 1;
  long max = 9000;
  unsigned long seed = 9000;

  std::uniform_int_distribution<long> unif(min, max);
  std::default_random_engine re(seed);

  if (columnStore) {
    for (long c = 0; c < width; c++) {
      for (long r = 0; r < height; r++) {
        matrix[IDX2C(r, c, height)] = unif(re);
      }
    }
  }
  else {
    for (long r = 0; r < height; r++) {
      for (long c = 0; c < width; c++) {
        matrix[r * width + c] = unif(re);
      }
    }
  }

}

size_t writeMatrix(std::string file, size_t matrixWidth, size_t matrixHeight, double *&matrix, bool silent) {
  if (!silent) {
    std::cout << "Writing " << file;
    std::cout.flush();
  }

  std::ofstream ofs(file, std::ios::binary);
  size_t numElems = matrixWidth * matrixHeight;

  ofs.write((const char *) matrix, numElems * sizeof(double));
  if (!ofs.good())
    std::cout << "Error writing file " << strerror(errno) << ": " << file << std::endl;
  ofs.flush();

  if (!silent)
    std::cout << " -- DONE" << std::endl;

  ofs.close();

  return numElems;
}

std::string generateFilename(std::string directory, MatrixType type, size_t blockRow, size_t blockCol, std::string suffix) {
  return std::string(
      directory + "/" + matrixTypeToString(type) + std::to_string(blockRow) + "-" + std::to_string(blockCol) + suffix
          + ".dat");
}
std::string generateDirectoryName(std::string basePath, size_t totalWidth, size_t totalHeight, size_t blockSize) {
  return std::string(basePath + "/" + std::to_string(totalWidth) + "x" + std::to_string(totalHeight) + "blksize"
                         + std::to_string(blockSize));
}

std::string generateDirectoryName(std::string basePath, size_t totalWidth, size_t totalHeight) {
  return std::string(basePath + "/" + std::to_string(totalWidth) + "x" + std::to_string(totalHeight));
}

int generateFullMatrixFile(std::string path, MatrixType type, size_t totalWidth, size_t totalHeight) {
  std::string blkDir = generateDirectoryName(path, totalWidth, totalHeight);

  if (create_dir(path) != 0)
    return -1;
  if (create_dir(blkDir) != 0)
    return -1;

  std::string matrixFile = std::string(blkDir + "/" + matrixTypeToString(type));

  std::cout << " Generating Matrix" << std::endl;
  double *matrix = allocMatrix(totalWidth, totalHeight);
  initMatrix(matrix, totalWidth, totalHeight, false);

  std::ofstream ofs(matrixFile, std::ios::binary);
  ofs.write((const char *) matrix, totalWidth * totalHeight * sizeof(double));
  ofs.flush();
  ofs.close();

  std::cout << " -- DONE" << std::endl;

  return 0;
}

int generateMatrixBlockFiles(std::string path,
                             MatrixType type,
                             size_t totalWidth,
                             size_t totalHeight,
                             size_t blockSize,
                             bool columnStore) {
  std::string blkDir = std::string(
      path + "/" + std::to_string(totalWidth) + "x" + std::to_string(totalHeight) + "blksize"
          + std::to_string(blockSize));
  int ret = create_dir(path);
  if (ret != 0 && ret != 1)
    return -1;

  ret = create_dir(blkDir);
  if (ret != 0 && ret != 1)
    return -1;

  std::string matrixDir = std::string(blkDir + "/" + matrixTypeToString(type));

  if (create_dir(matrixDir) != 0)
    return -1;

  size_t numBlocksWidth = totalWidth / blockSize;
  size_t numBlocksHeight = totalHeight / blockSize;

  // leftover
  if (totalWidth % blockSize != 0)
    numBlocksWidth++;

  if (totalHeight % blockSize != 0)
    numBlocksHeight++;

  std::cout << " Generating Matrix " << totalHeight << "x" << totalWidth << " blockSize: " << blockSize
            << " numBlocksWidth: " << numBlocksWidth << " numBlocksHeight: " << numBlocksHeight
            << " -- Allocating and Initializing...";
  std::cout.flush();

  double *matrix = allocMatrix(blockSize, blockSize);
  initMatrix(matrix, blockSize, blockSize, columnStore);
  std::cout << " -- DONE" << std::endl;

  for (size_t blockRow = 0; blockRow < numBlocksHeight; blockRow++) {
    size_t matrixHeight =
        ((blockRow == numBlocksHeight - 1 && totalHeight % blockSize != 0) ? totalHeight % blockSize : blockSize);
    for (size_t blockCol = 0; blockCol < numBlocksWidth; blockCol++) {
      size_t matrixWidth =
          ((blockCol == numBlocksWidth - 1 && totalWidth % blockSize != 0) ? totalWidth % blockSize : blockSize);
      std::string fileName = std::string(matrixDir + "/" + std::to_string(blockRow) + "_" + std::to_string(blockCol));
      std::cout << "Writing " << fileName << " - blockRow: " << blockRow << " blockCol: " << blockCol << ": "
                << matrixHeight << "x" << matrixWidth;
      std::cout.flush();

      std::ofstream ofs(fileName, std::ios::binary);
      ofs.write((const char *) matrix, matrixWidth * matrixHeight * sizeof(double));
      ofs.flush();
      ofs.close();

      std::cout << " -- DONE" << std::endl;
    }
  }

  delete[] matrix;

  return 0;
}

double *readMatrix(std::string path,
                   MatrixType type,
                   size_t totalWidth,
                   size_t totalHeight,
                   size_t blockSize,
                   size_t blockRow,
                   size_t blockCol,
                   std::string suffix) {
  std::string directory = generateDirectoryName(path,
                                                totalWidth,
                                                totalHeight,
                                                blockSize);
  std::string fileName = generateFilename(directory,
                                          type,
                                          blockRow,
                                          blockCol,
                                          suffix);

  size_t numBlocksWidth = totalWidth / blockSize;
  size_t numBlocksHeight = totalHeight / blockSize;

  // leftover
  if (totalWidth % blockSize != 0)
    numBlocksWidth++;

  if (totalHeight % blockSize != 0)
    numBlocksHeight++;

  size_t matrixWidth =
      ((blockCol == numBlocksWidth - 1 && totalWidth % blockSize != 0) ? totalWidth % blockSize : blockSize);
  size_t matrixHeight =
      ((blockRow == numBlocksHeight - 1 && totalHeight % blockSize != 0) ? totalHeight % blockSize : blockSize);

  double *matrix = allocMatrix(matrixWidth, matrixHeight);

  std::ifstream ifs(fileName, std::ios::binary);
  if (!ifs.good()) {
    std::cout << "Error reading file " << strerror(errno) << ": " << fileName << std::endl;
  }
  ifs.read((char *) matrix, matrixWidth * matrixHeight * sizeof(double));
  ifs.close();
  return matrix;
}

size_t readMatrix(std::string path,
                  MatrixType type,
                  size_t totalWidth,
                  size_t totalHeight,
                  size_t blockSize,
                  size_t blockRow,
                  size_t blockCol,
                  double *&matrix,
                  std::string suffix) {
  std::string directory = generateDirectoryName(path,
                                                totalWidth,
                                                totalHeight,
                                                blockSize);

  std::string fileName = generateFilename(directory,
                                          type,
                                          blockRow,
                                          blockCol,
                                          suffix);
  size_t numBlocksWidth = totalWidth / blockSize;
  size_t numBlocksHeight = totalHeight / blockSize;

  // leftover
  if (totalWidth % blockSize != 0)
    numBlocksWidth++;

  if (totalHeight % blockSize != 0)
    numBlocksHeight++;

  size_t matrixWidth =
      ((blockCol == numBlocksWidth - 1 && totalWidth % blockSize != 0) ? totalWidth % blockSize : blockSize);
  size_t matrixHeight =
      ((blockRow == numBlocksHeight - 1 && totalHeight % blockSize != 0) ? totalHeight % blockSize : blockSize);

  std::ifstream ifs(fileName, std::ios::binary);
  if (!ifs.good()) {
    std::cout << "Error reading file " << strerror(errno) << ": " << fileName << std::endl;
  }

  size_t numElems = matrixWidth * matrixHeight;
  ifs.read((char *) matrix, numElems * sizeof(double));
  ifs.close();

  return numElems;
}

bool checkMatrixFiles(std::string path, MatrixType type, size_t totalWidth, size_t totalHeight) {
  std::string dir = generateDirectoryName(path, totalWidth, totalHeight);
  std::string fileName = dir + "/" + matrixTypeToString(type);

  if (!has_dir(path)) {
    std::cout << "Unable to find directory: " << path << std::endl;
    return false;
  }

  if (!has_dir(dir)) {
    std::cout << "Unable to find directory: " << dir << std::endl;
    return false;
  }

  if (!has_file(fileName)) {
    std::cout << "Unable to find file: " << fileName << std::endl;
    return false;
  }

  return true;

}

bool checkMatrixBlockFiles(std::string path, MatrixType type, size_t totalWidth, size_t totalHeight, size_t blockSize) {
  std::string blkDir = generateDirectoryName(path, totalWidth, totalHeight, blockSize);
  std::string matrixDir = std::string(blkDir + "/" + matrixTypeToString(type));

  // check dir path
  if (!has_dir(path)) {
    std::cout << "Unable to find directory: " << path << std::endl;
    return false;
  }

  // check dir blkDir
  if (!has_dir(blkDir)) {
    std::cout << "Unable to find directory: " << blkDir << std::endl;
    return false;
  }

  // check dir matrixDir
  if (!has_dir(matrixDir)) {
    std::cout << "Unable to find directory: " << matrixDir << std::endl;
    return false;
  }

  size_t numBlocksWidth = totalWidth / blockSize;
  size_t numBlocksHeight = totalHeight / blockSize;

  // leftover
  if (totalWidth % blockSize != 0)
    numBlocksWidth++;

  if (totalHeight % blockSize != 0)
    numBlocksHeight++;

  for (size_t blockRow = 0; blockRow < numBlocksHeight; blockRow++) {
    for (size_t blockCol = 0; blockCol < numBlocksWidth; blockCol++) {
      std::string fileName = std::string(matrixDir + "/" + std::to_string(blockRow) + "_" + std::to_string(blockCol));

      // check file
      if (!has_file(fileName)) {
        std::cout << "Unable to find file: " << fileName << std::endl;
        return false;
      }
    }
  }

  return true;
}

void checkAndValidateMatrixBlockFiles(std::string directory,
                                      size_t widthA,
                                      size_t heightA,
                                      size_t widthB,
                                      size_t heightB,
                                      size_t blockSize,
                                      bool columnStore) {
  if (!has_dir(directory))
    create_dir(directory);

  bool checkA = checkMatrixBlockFiles(directory, MatrixType::MatrixA, widthA, heightA, blockSize);

  if (!checkA) {
    if (generateMatrixBlockFiles(directory, MatrixType::MatrixA, widthA, heightA, blockSize, columnStore))
      exit(-1);
  }

  bool checkB = checkMatrixBlockFiles(directory, MatrixType::MatrixB, widthB, heightB, blockSize);

  if (!checkB) {
    if (generateMatrixBlockFiles(directory, MatrixType::MatrixB, widthB, heightB, blockSize, columnStore) != 0)
      exit(-1);
  }
}

void checkAndValidateMatrixFiles(std::string directory, size_t widthA, size_t heightA, size_t widthB, size_t heightB) {
  if (!has_dir(directory))
    create_dir(directory);

  bool checkA = checkMatrixFiles(directory, MatrixType::MatrixA, widthA, heightA);
  if (!checkA) {
    if (generateFullMatrixFile(directory, MatrixType::MatrixA, widthA, heightA))
      exit(-1);
  }

  bool checkB = checkMatrixFiles(directory, MatrixType::MatrixB, widthB, heightB);

  if (!checkB) {
    if (generateFullMatrixFile(directory, MatrixType::MatrixB, widthB, heightB))
      exit(-1);
  }
}