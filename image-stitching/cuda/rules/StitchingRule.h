//
// Created by tjb3 on 11/18/15.
//

#include <tile-grid.hpp>
#include <htgs/api/IRule.hpp>
#include "../data/PCIAMData.h"
#include "../data/FFTData.h"

#ifndef HTGS_STITCHINGRULE_H
#define HTGS_STITCHINGRULE_H
enum State {
  NONE = 0, IN_FLIGHT = 1, COMPLETED = 2
};

class StitchingRule: public htgs::IRule<FFTData, PCIAMData> {


 public:

  StitchingRule(TileGrid<ImageStitching::CUDAImageTile> *grid) {
    this->grid = grid;
    this->width = grid->getExtentWidth();
    this->height = grid->getExtentHeight();
    this->fftState = allocateState();
    this->pciamWestState = allocateState();
    this->pciamNorthState = allocateState();
    this->fftDataGrid = allocateFFTDataGrid();

    initStateArray(this->fftState);
    initStateArray(this->pciamWestState);
    initStateArray(this->pciamNorthState);
    initFFTDataGrid(this->fftDataGrid);
  }

  ~StitchingRule() {
    deallocState(this->fftState, this->height);
    deallocState(this->pciamNorthState, this->height);
    deallocState(this->pciamWestState, this->height);
    deAllocFFTDataGrid();
  }

  bool isRuleTerminated(int pipelineId) { return false; }


  std::string getName() { return "StitchingRule"; }

  void applyRule(std::shared_ptr<FFTData> data, int pipelineId) {
    ImageStitching::CUDAImageTile *tile = data->getTile();

    DEBUG("Stitching rule received tile: " << tile->getFilename());

    int r = tile->getRowIdx() - this->grid->getStartRow();
    int c = tile->getColIdx() - this->grid->getStartCol();

    this->fftState[r][c] = COMPLETED;

    this->fftDataGrid[r][c] = data;

    // PCIAM_DIRECTION_NORTH
    if (r > 0 && this->fftState[r - 1][c] == COMPLETED && pciamNorthState[r][c] == NONE) {
      pciamNorthState[r][c] = IN_FLIGHT;
      this->addResult(new PCIAMData(data, this->fftDataGrid[r - 1][c], PCIAM_DIRECTION_NORTH, data->getOrder()));
    }

    // PCIAM_DIRECTION_WEST
    if (c > 0 && this->fftState[r][c - 1] == COMPLETED && pciamWestState[r][c] == NONE) {
      pciamWestState[r][c] = IN_FLIGHT;
      this->addResult(new PCIAMData(data, this->fftDataGrid[r][c - 1], PCIAM_DIRECTION_WEST, data->getOrder()));

    }

    // South
    if (r < height - 1 && fftState[r + 1][c] == COMPLETED && pciamNorthState[r + 1][c] == NONE) {
      pciamNorthState[r + 1][c] = IN_FLIGHT;
      this->addResult(new PCIAMData(this->fftDataGrid[r + 1][c], data, PCIAM_DIRECTION_NORTH, data->getOrder()));
    }

    //East
    if (c < width - 1 && fftState[r][c + 1] == COMPLETED && pciamWestState[r][c + 1] == NONE) {
      pciamWestState[r][c + 1] = IN_FLIGHT;
      this->addResult(new PCIAMData(this->fftDataGrid[r][c + 1], data, PCIAM_DIRECTION_WEST, data->getOrder()));
    }

//        std::cout << "FFT State width : " << this->width << " height: " << this->height << std::endl;
//        printArr(this->fftState);
//        std::cout << "PCIAM NORTH State width : " << this->width << " height: " << this->height << std::endl;
//        printArr(this->pciamNorthState);
//        std::cout << "PCIAM WEST State width : " << this->width << " height: " << this->height << std::endl;
//        printArr(this->pciamWestState);

  }

  void shutdownRule(int pipelineId) {

  }

 private:
  void printArr(State **arr) {
    std::cout << std::endl << "---------------------------------------------------" << std::endl;

    for (int r = 0; r < height; r++) {
      for (int c = 0; c < width; c++) {
        std::cout << std::to_string(arr[r][c]) + " ";
      }
      std::cout << std::endl;
    }


    std::cout << "---------------------------------------------------" << std::endl;
  }

  std::shared_ptr<FFTData> **allocateFFTDataGrid() {
    std::shared_ptr<FFTData> **fftDataGrid = new std::shared_ptr<FFTData> *[height];
    for (int r = 0; r < height; r++) {
      fftDataGrid[r] = new std::shared_ptr<FFTData>[width];
    }
    return fftDataGrid;
  }

  void deAllocFFTDataGrid() {
    for (int r = 0; r < height; r++) {
      delete[] fftDataGrid[r];
    }
    delete[] fftDataGrid;

  }

  void initFFTDataGrid(std::shared_ptr<FFTData> **grid) {
    for (int r = 0; r < height; r++)
      for (int c = 0; c < width; c++)
        grid[r][c] = nullptr;
  }


  void initStateArray(State **arr) {
    for (int r = 0; r < height; r++)
      for (int c = 0; c < width; c++)
        arr[r][c] = NONE;
  }

  State **allocateState() {
    State **arr = new State *[height];
    for (int i = 0; i < height; i++)
      arr[i] = new State[width];
    return arr;
  }

  void deallocState(State **arr, int height) {
    for (int i = 0; i < height; i++)
      delete[] arr[i];
    delete[] arr;
  }

  State **fftState;
  State **pciamWestState;
  State **pciamNorthState;
  std::shared_ptr<FFTData> **fftDataGrid;

  int width;
  int height;

  TileGrid<ImageStitching::CUDAImageTile> *grid;

};

#endif //HTGS_STITCHINGRULE_H

