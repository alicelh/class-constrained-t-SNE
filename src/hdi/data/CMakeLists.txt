include_directories ("${CMAKE_SOURCE_DIR}")

file(GLOB HeaderFiles *.h)
file(GLOB SourceFiles *.cpp)

add_library(hdidata ${HeaderFiles} ${SourceFiles} )


