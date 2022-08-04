cmake_minimum_required(VERSION 3.23)
project(PerlinNoise CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(PerlinNoise main.cu Managed.cuh ppm.h)

set_target_properties(PerlinNoise PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
