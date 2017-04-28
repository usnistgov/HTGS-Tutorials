
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 1/15/16.
//

#include <ostream>
#include <string.h>
#include <iostream>
#include <sys/stat.h>
#include "util-filesystem.h"

int create_dir(std::string path) {
#ifdef _WIN32
  std::wstring wpath = std::wstring(path.begin(), path.end());
  std::wcout << " Creating folder " << wpath << std::endl;
  int val = _wmkdir(wpath.c_str());
#else
  int val = mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif

  if (val == 0) {
    std::cout << "directory: " << path << " created successfully " << std::endl;
    return 0;
  }
  else {
    if (errno == EEXIST)
      return 0;

    std::cout << "Unable to create directory " << path << ": " << strerror(errno)
              << std::endl; // << val << " " << path << std::endl;
    return val;
  }

}

bool has_dir(std::string path) {
  struct stat info;
  if (stat(path.c_str(), &info) != 0) {
    std::cout << "cannot access " << path << std::endl;
    return false;
  }
  else if (info.st_mode & S_IFDIR) {
    return true;
  }

  std::cout << path << " is not a directory" << std::endl;

  return false;
}

bool has_file(std::string filePath) {
  struct stat info;
  if (stat(filePath.c_str(), &info) != 0) {
    std::cout << "cannot access " << filePath << std::endl;
    return false;
  }
  return true;
}
