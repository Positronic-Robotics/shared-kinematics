cmake_minimum_required(VERSION 3.18)

project(shared_kinematics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SHARED_KINEMATICS_PYTHON "Build the Python module" ON)
option(SHARED_KINEMATICS_TESTS "Build the C++ tests and the timing run" OFF)

# Eigen: an installed package, else the headers at EIGEN3_INCLUDE_DIR, else Eigen 3.4.0 from its release archive.
find_package(Eigen3 3.3 CONFIG QUIET)
if (NOT TARGET Eigen3::Eigen)
  if (NOT EIGEN3_INCLUDE_DIR AND DEFINED ENV{EIGEN3_INCLUDE_DIR})
    set(EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}")
  endif()
  if (NOT EIGEN3_INCLUDE_DIR)
    include(FetchContent)
    if (POLICY CMP0135)
      cmake_policy(SET CMP0135 NEW)
    endif()
    # SOURCE_SUBDIR names no CMakeLists.txt: the archive gives the headers and builds nothing.
    FetchContent_Declare(eigen
      URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
      URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
      SOURCE_SUBDIR headers-only)
    FetchContent_MakeAvailable(eigen)
    set(EIGEN3_INCLUDE_DIR "${eigen_SOURCE_DIR}")
  endif()
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

add_library(shared_kinematics INTERFACE)
add_library(shared_kinematics::shared_kinematics ALIAS shared_kinematics)
target_include_directories(shared_kinematics INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(shared_kinematics INTERFACE Eigen3::Eigen)
target_compile_features(shared_kinematics INTERFACE cxx_std_17)

if (SHARED_KINEMATICS_PYTHON)
  set(PYBIND11_FINDPYTHON NEW)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core src/bindings.cpp)
  target_link_libraries(_core PRIVATE shared_kinematics)
  install(TARGETS _core LIBRARY DESTINATION shared_kinematics)
  # The headers ship in the wheel, so a C++ driver can build against the same core.
  install(DIRECTORY include/ DESTINATION shared_kinematics/include)
endif()

if (SHARED_KINEMATICS_TESTS)
  enable_testing()
  foreach (name test_guard bench_guard)
    add_executable(${name} tests/${name}.cpp)
    target_include_directories(${name} PRIVATE tests)
    target_link_libraries(${name} PRIVATE shared_kinematics)
    target_compile_options(${name} PRIVATE -Wall -Wextra -Werror)
  endforeach()
  add_test(NAME test_guard COMMAND test_guard)
endif()
