set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB HeaderFiles *.h)
file(GLOB SourceFiles *.cpp)

add_library(hdiutils ${HeaderFiles} ${SourceFiles} )

install (TARGETS hdiutils DESTINATION bin/hdi)
