cmake_minimum_required(VERSION 3.5)
project(main)

set (CMAKE_CXX_FLAGS "-std=c++11 -pthread")

add_executable(main main.cc)
find_package(Easy3D REQUIRED)

target_link_libraries(main easy3d::viewer) # request necessary Easy3D modules (add more if needed, e.g., viewer, algo)

