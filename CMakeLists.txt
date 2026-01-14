cmake_minimum_required(VERSION 3.20)
project(untitled LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenMP REQUIRED)

# Trova CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(CudaConv CudaConv.cu)

#find_package(CURL REQUIRED)

target_link_libraries(CudaConv ${OpenCV_LIBS} ${CUDA_LIBRARIES})
#target_link_libraries(CudaConv ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CURL_LIBRARIES})

# Imposta le proprietà CUDA
set_target_properties(CudaConv PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "60;70;75;80;86")

