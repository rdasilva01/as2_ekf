cmake_minimum_required(VERSION 3.5)
set(PROJECT_NAME ekf)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(${PROJECT_NAME})

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Set Release as default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Find dependencies
set(PROJECT_DEPENDENCIES
  Eigen3
)

foreach(DEPENDENCY ${PROJECT_DEPENDENCIES})
  find_package(${DEPENDENCY} REQUIRED)
  message(STATUS "Found ${DEPENDENCY}: ${${DEPENDENCY}_FOUND}")
endforeach()

message(STATUS "Using Eigen3: ${EIGEN3_INCLUDE_DIRS}")

include_directories(
  include
  include/${PROJECT_NAME}
  ${EIGEN3_INCLUDE_DIRS}
)

set(SOURCE_H_FILES
  include/${PROJECT_NAME}/ekf_c_code.h
  include/${PROJECT_NAME}/ekf_datatype.hpp
  include/${PROJECT_NAME}/ekf_wrapper.hpp
)

set(SOURCE_CPP_FILES
  src/ekf_c_code.cpp
  src/ekf_datatype.cpp
  src/ekf_wrapper.cpp
)

# Create the library
add_library(${PROJECT_NAME} SHARED ${SOURCE_CPP_FILES})

# Export library headers
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# # Link the library with dependencies
# target_link_libraries(${PROJECT_NAME}
#   ${PROJECT_DEPENDENCIES}
#   ${Eigen3_LIBRARIES}
# )

# Add executables for each example
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
