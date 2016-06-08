//
// Created by tjb3 on 1/15/16.
//

#include <random>
#include <string.h>

#include "util-matrix.h"
#include "util-filesystem.h"

double *allocMatrix(int width, int height) {
  double *matrix = new double[height * width];
  return matrix;
}

void initMatrix(double *matrix, int width, int height) {
  long min = 1;
  long max = 9000;

  std::uniform_int_distribution<int> unif(min, max);
  std::default_random_engine re;

  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {
      matrix[r*width+c] = unif(re);
    }
  }

}

size_t writeMatrix(std::string file, int matrixWidth, int matrixHeight, double *&matrix, bool silent) {
  if (!silent) {
    std::cout << "Writing " << file;
    std::cout.flush();
  }

    std::ofstream ofs (file, std::ios::binary);
    size_t numElems = (size_t)matrixWidth*(size_t)matrixHeight;

    ofs.write((const char *) matrix, numElems * sizeof(double));
    if (!ofs.good())
        std::cout << "Error writing file " << strerror(errno) << ": " << file << std::endl;
    ofs.flush();

  if (!silent)
    std::cout << " -- DONE" << std::endl;

    ofs.close();

  return numElems;
}

std::string generateFilename(std::string directory, MatrixType type, int blockRow, int blockCol, std::string suffix) {
  return std::string(
      directory + "/" + matrixTypeToString(type) + std::to_string(blockRow) + "-" + std::to_string(blockCol) + suffix
          + ".dat");
}
std::string generateDirectoryName(std::string basePath, int totalWidth, int totalHeight, int blockSize) {
  return std::string(basePath + "/" + std::to_string(totalWidth) + "x" + std::to_string(totalHeight) + "blksize"
                         + std::to_string(blockSize));
}

int generateFullMatrixFile(std::string path, MatrixType type, int totalWidth, int totalHeight) {
  std::string blkDir = std::string(path + "/" + std::to_string(totalWidth) + "x"  + std::to_string(totalHeight));
  if (create_dir(path) != 0)
    return -1;
  if (create_dir(blkDir) != 0)
    return -1;

  std::string matrixFile = std::string(blkDir + "/" + matrixTypeToString(type));


  std::cout << " Generating Matrix" << std::endl;
  double *matrix = allocMatrix(totalWidth, totalHeight);
  initMatrix(matrix, totalWidth, totalHeight);

  std::ofstream ofs(matrixFile, std::ios::binary);
  ofs.write((const char *) matrix, totalWidth * totalHeight* sizeof(double));
  ofs.flush();
  ofs.close();

  std::cout << " -- DONE" << std::endl;

  return 0;
}

int generateMatrixBlockFiles(std::string path, MatrixType type, int totalWidth, int totalHeight, int blockSize) {
  std::string blkDir = std::string(path + "/" + std::to_string(totalWidth) + "x"  + std::to_string(totalHeight) + "blksize" + std::to_string(blockSize));
  int ret = create_dir(path);
  if (ret != 0 && ret != 1)
    return -1;

  ret = create_dir(blkDir);
  if (ret != 0 && ret != 1)
    return -1;

  std::string matrixDir = std::string(blkDir + "/" + matrixTypeToString(type));

  if (create_dir(matrixDir) != 0)
    return -1;


  int numBlocksWidth = totalWidth / blockSize;
  int numBlocksHeight = totalHeight / blockSize;

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
  initMatrix(matrix, blockSize, blockSize);
  std::cout << " -- DONE" << std::endl;

  for (int blockRow = 0; blockRow < numBlocksHeight; blockRow++) {
    int matrixHeight =
        ((blockRow == numBlocksHeight - 1 && totalHeight % blockSize != 0) ? totalHeight % blockSize : blockSize);
    for (int blockCol = 0; blockCol < numBlocksWidth; blockCol++) {
      int matrixWidth =
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
                  int totalWidth,
                  int totalHeight,
                  int blockSize,
                  int blockRow,
                  int blockCol,
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

  int numBlocksWidth = totalWidth / blockSize;
  int numBlocksHeight = totalHeight / blockSize;

  // leftover
  if (totalWidth % blockSize != 0)
    numBlocksWidth++;

  if (totalHeight % blockSize != 0)
    numBlocksHeight++;

  int matrixWidth =
      ((blockCol == numBlocksWidth - 1 && totalWidth % blockSize != 0) ? totalWidth % blockSize : blockSize);
  int matrixHeight =
      ((blockRow == numBlocksHeight - 1 && totalHeight % blockSize != 0) ? totalHeight % blockSize : blockSize);

  double *matrix = allocMatrix(matrixWidth, matrixHeight);

  std::ifstream ifs(fileName, std::ios::binary);
  if (!ifs.good())
  {
      std::cout << "Error reading file " << strerror(errno) << ": " << fileName << std::endl;
  }
  ifs.read((char *) matrix, matrixWidth * matrixHeight * sizeof(double));
  ifs.close();
  return matrix;
}

size_t readMatrix(std::string path,
                  MatrixType type,
                  int totalWidth,
                  int totalHeight,
                  int blockSize,
                  int blockRow,
                  int blockCol,
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
  int numBlocksWidth = totalWidth / blockSize;
  int numBlocksHeight = totalHeight / blockSize;

  // leftover
  if (totalWidth % blockSize != 0)
    numBlocksWidth++;

  if (totalHeight % blockSize != 0)
    numBlocksHeight++;

  int matrixWidth =
      ((blockCol == numBlocksWidth - 1 && totalWidth % blockSize != 0) ? totalWidth % blockSize : blockSize);
  int matrixHeight =
      ((blockRow == numBlocksHeight - 1 && totalHeight % blockSize != 0) ? totalHeight % blockSize : blockSize);

  std::ifstream ifs(fileName, std::ios::binary);
  if (!ifs.good())
  {
      std::cout << "Error reading file " << strerror(errno) << ": " << fileName << std::endl;
  }

  size_t numElems = matrixWidth *matrixHeight;
  ifs.read((char *) matrix, numElems * sizeof(double));
  ifs.close();

  return numElems;
}

bool checkMatrixBlockFiles(std::string path, MatrixType type, int totalWidth, int totalHeight, int blockSize) {
  std::string blkDir = std::string(
      path + "/" + std::to_string(totalWidth) + "x" + std::to_string(totalHeight) + "blksize"
          + std::to_string(blockSize));
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

  int numBlocksWidth = totalWidth / blockSize;
  int numBlocksHeight = totalHeight / blockSize;

  // leftover
  if (totalWidth % blockSize != 0)
    numBlocksWidth++;

  if (totalHeight % blockSize != 0)
    numBlocksHeight++;

  for (int blockRow = 0; blockRow < numBlocksHeight; blockRow++) {
    for (int blockCol = 0; blockCol < numBlocksWidth; blockCol++) {
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
