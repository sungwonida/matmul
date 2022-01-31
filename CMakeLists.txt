cmake_minimum_required (VERSION 3.10)
project (matmul LANGUAGES CXX CUDA)
set (CMAKE_VERBOSE_MAKEFILE true)
set (CMAKE_CXX_STANDARD 14)

# set (OpenCV_DIR "$ENV{HOME}/.local/opencv-4.5.1/lib")
# find_package (OpenCV 4 REQUIRED PATHS "$ENV{HOME}/.local/opencv-4.5.1" COMPONENTS world)
find_package (OpenCV 4 REQUIRED COMPONENTS world)

add_executable (matmul mat_mul.cpp mat_mul.cu)
target_link_libraries (matmul ${OpenCV_LIBS})
set_target_properties(matmul
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  -ccbin g++
  -m64
  -gencode arch=compute_86,code=sm_86  # Modify to what you have
  --use_fast_math
  --relocatable-device-code=true
  >)
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
  -march=skylake-avx512  # Comment out if you don't have AVX512
  -mavx2 -mfma
  >)
