cmake_minimum_required(VERSION 3.1.0)
project(cstsne)
set (CMAKE_CXX_STANDARD 20)


# Find pybind11
add_subdirectory(pybind11)
add_subdirectory(hdi/data)
add_subdirectory(hdi/dimensionality_reduction)
add_subdirectory(hdi/utils)



pybind11_add_module(cstsne cstsnebind.cpp)
target_link_libraries(cstsne PUBLIC hdidata hdidimensionalityreduction hdiutils)
