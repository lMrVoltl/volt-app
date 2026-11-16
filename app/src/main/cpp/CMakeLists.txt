cmake_minimum_required(VERSION 3.22.1)
project(volt_jni)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -march=armv8.2-a+dotprod+fp16+i8mm -ffast-math")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -DNDEBUG -march=armv8.2-a+dotprod+fp16+i8mm -ffast-math")

set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)
add_subdirectory(llama.cpp/tools/mtmd)

add_library(volt_jni SHARED volt_jni.cpp)
target_include_directories(volt_jni PRIVATE llama.cpp/tools/mtmd)
target_link_libraries(volt_jni llama mtmd android log)
