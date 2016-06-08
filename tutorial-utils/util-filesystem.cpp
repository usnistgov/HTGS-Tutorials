//
// Created by tjb3 on 1/15/16.
//

#include <ostream>
#include <string.h>
#include <iostream>
#include <sys/stat.h>
#include "util-filesystem.h"

int create_dir(std::string path) {
#ifdef __linux__
  int val = mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#else
  std::wstring wpath = std::wstring(path.begin(), path.end());
  std::wcout << " Creating folder " << wpath << std::endl;
  int val = _wmkdir(wpath.c_str());
#endif


  if (val == 0) {
    std::cout << "directory: " << path << " created successfully " << std::endl;
    return 0;
  }
  else {
    if (errno == EEXIST)
      return 0;

    std::cout << "Unable to create directory " << path <<": " << strerror(errno) << std::endl; // << val << " " << path << std::endl;
    return val;
  }

}

bool has_dir(std::string path) {
  struct stat info;
  if (stat(path.c_str(), &info) != 0)
  {
    std::cout << "cannot access " << path << std::endl;
    return false;
  }
  else if (info.st_mode & S_IFDIR)
  {
    return true;
  }

  std::cout << path << " is not a directory" << std::endl;

  return false;
}

bool has_file(std::string filePath) {
  struct stat info;
  if (stat(filePath.c_str(), &info) != 0)
  {
    std::cout << "cannot access " << filePath << std::endl;
    return false;
  }
  return true;
}