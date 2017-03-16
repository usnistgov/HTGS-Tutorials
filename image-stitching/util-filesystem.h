//
// Created by tjb3 on 1/29/16.
//

#ifndef CMAKE_IMAGE_STITCHING_UTIL_FILESYSTEM_H
#define CMAKE_IMAGE_STITCHING_UTIL_FILESYSTEM_H

#include <limits.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <assert.h>

bool _dirExists(const char *path)
{
    struct stat sb;
    return stat(path, &sb) == 0 && S_ISDIR(sb.st_mode);
}

#ifdef __linux__

int _mkdir(char* file_path, mode_t mode) {
    assert(file_path && *file_path);
    char* p;
    for (p=strchr(file_path+1, '/'); p; p=strchr(p+1, '/')) {
        *p='\0';
        if (mkdir(file_path, mode)==-1) {
            if (errno!=EEXIST) { *p='/'; return -1; }
        }
        *p='/';
    }
    return 0;
}

#else
int _mkdir(wchar_t* file_path, mode_t mode) {
    assert(file_path && *file_path);
    char* p;
    for (p=strchr(file_path+1, '/'); p; p=strchr(p+1, '/')) {
        *p='\0';
        if (_wmkdir(file_path, mode)==-1) {
            if (errno!=EEXIST) { *p='/'; return -1; }
        }
        *p='/';
    }
    return 0;
}

#endif

#endif //CMAKE_IMAGE_STITCHING_UTIL_FILESYSTEM_H
