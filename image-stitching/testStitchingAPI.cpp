//
// Created by tjb3 on 11/16/15.
//

#include <fftw-image-tile.h>
#include <tile-grid.hpp>
#include <tile-grid-traverser.hpp>
#include <htgs/core/memory/MemoryPool.hpp>
#include <htgs/api/MemoryData.hpp>
#include "cpu/memory/FFTWMemory.h"
#include "cpu/memory/FFTMemoryRule.h"


using namespace htgs;

namespace is = ImageStitching;

int main() {

  std::cout << "Testing Stitching API test" << std::endl;

  int startRow = 0;
  int startCol = 0;
  int extentWidth = 5;
  int extentHeight = 5;

  std::string path("/home/tjb3/datasets/image-stitching/1h_Wet_10Perc");

  TileGrid<is::FFTWImageTile> *grid = new TileGrid<is::FFTWImageTile>(startRow,
                                                                      startCol,
                                                                      extentWidth,
                                                                      extentHeight,
                                                                      23,
                                                                      30,
                                                                      GridOrigin::UpperRight,
                                                                      GridNumbering::Column,
                                                                      1,
                                                                      path,
                                                                      "KB_2012_04_13_1hWet_10Perc_IR_0{pppp}.tif",
                                                                      is::ImageTileType::FFTW);

  is::FFTWImageTile *tile = grid->getSubGridTilePtr(0, 0);
  tile->readTile();
  is::FFTWImageTile::initPlans(tile->getWidth(), tile->getHeight(), 0x42, true, "test.dat");
  is::FFTWImageTile::savePlan("test.dat");

  TileGridTraverser<is::FFTWImageTile> *traverser = createTraverser(grid, Traversal::DiagonalTraversal);

  MemoryPool<fftw_t *> *memoryPool = new MemoryPool<fftw_t *>(90);

  FFTWMemory *memoryAllocator = new FFTWMemory(500);
  std::shared_ptr<FFTWMemory> memAlloc(memoryAllocator);
//    IMemoryAllocator<fftw_complex *> *test = new IMemoryAllocator<fftw_complex *>();

  MemoryData<fftw_t *> *fftwMemory = new MemoryData<fftw_t *>(memAlloc);
//    Memory<fftw_complex *> *fftwMemory = new Memory<fftw_complex *>(test, 500);

  memoryPool->fillPool(fftwMemory, 0, true);

  std::shared_ptr<MemoryData<fftw_t *>> myMem = memoryPool->getMemory();
  DEBUG_VERBOSE("Memory = " << myMem.get());

  FFTMemoryRule *rule = new FFTMemoryRule(tile,
                                          grid->getExtentWidth(),
                                          grid->getExtentHeight(),
                                          grid->getStartRow(),
                                          grid->getStartCol());

  myMem->setMemoryReleaseRule(rule);

  std::cout << "Testing memory FFT release rule" << std::endl;

  int count = 0;
  while (!myMem->canReleaseMemory()) {
    myMem->memoryUsed();
    count++;
  }

  std::cout << "Can release memory after " << count << " iterations" << std::endl;

  std::cout << "Finished Stitching API test" << std::endl;
}