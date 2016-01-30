cmake_minimum_required(VERSION 3.3)
project(Projekt)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
        main.cpp
        dijkstraMpi.cpp
        shortestPath.cpp
    )

#set(CMAKE_C_COMPILER mpicc)
#set(CMAKE_CXX_COMPILER mpicxx)


add_executable(Projekt ${SOURCE_FILES})