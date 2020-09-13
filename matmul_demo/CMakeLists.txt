cmake_minimum_required (VERSION 3.10)
project (matmul_demo)
set (CMAKE_VERBOSE_MAKEFILE true)

set (CMAKE_CXX_STANDARD 14)
message("C++ compiler flags: ${CMAKE_CXX_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")

if (UNIX)
  set (OpenCV_DIR "$ENV{HOME}/.local/lib")
endif (UNIX)
if (WIN32)
  set (OpenCV_DIR "D:/opencv/build/x64/vc15/lib")
endif (WIN32)

message ("OpenCV_DIR: ${OpenCV_DIR}")
find_package (OpenCV REQUIRED COMPONENTS world)
message ("OpenCV_LIBS: ${OpenCV_LIBS}")

# include_directories (${OpenCV_INCLUDE_DIRS})
add_executable (matmul_demo mat_mul.cpp)
target_link_libraries (matmul_demo ${OpenCV_LIBS})
