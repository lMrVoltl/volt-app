cmake_minimum_required(VERSION 3.22.1)
project(volt_jni)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -DNDEBUG")

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

file(GLOB MTMD_SRCS "llama.cpp/tools/mtmd/*.cpp")
list(FILTER MTMD_SRCS EXCLUDE REGEX ".*cli.*")

add_library(volt_jni SHARED volt_jni.cpp ${MTMD_SRCS})
target_include_directories(volt_jni PRIVATE llama.cpp/tools/mtmd llama.cpp/vendor)
target_link_libraries(volt_jni llama android log)
