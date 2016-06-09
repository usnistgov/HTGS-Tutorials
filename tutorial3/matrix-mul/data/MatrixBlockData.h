//
// Created by tjb3 on 2/23/16.
//

#ifndef HTGS_MATRIXBLOCKDATA_H
#define HTGS_MATRIXBLOCKDATA_H

#include <htgs/api/MemoryData.hpp>

typedef std::shared_ptr<htgs::MemoryData<double *>> MatrixMemoryData_t;

template <class T>
class MatrixBlockData : public htgs::IData
{
 public:


  MatrixBlockData(const std::shared_ptr<MatrixRequestData> &request,
                  const T &matrixData,
                  int matrixWidth,
                  int matrixHeight) :
  request(request), matrixData(matrixData), matrixWidth(matrixWidth), matrixHeight(matrixHeight) { }

  const std::shared_ptr<MatrixRequestData> &getRequest() const {
    return request;
  }
  const T &getMatrixData() const {
    return matrixData;
  }
  int getMatrixWidth() const {
    return matrixWidth;
  }
  int getMatrixHeight() const {
    return matrixHeight;
  }

 private:
  std::shared_ptr<MatrixRequestData> request;
  T matrixData;
  int matrixWidth;
  int matrixHeight;
};
#endif //HTGS_MATRIXBLOCKDATA_H
