CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

FIND_PACKAGE(deal.II 9.3.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

IF(NOT DEAL_II_WITH_P4EST) # keep in one line
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_P4EST = ON
It was set to OFF in your installation but is required for this program."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

SET( TARGET_SRC dummy.cc)

PROJECT(data_locality)

ADD_LIBRARY(data_locality  ${TARGET_SRC})

# Define custom targets to easily switch the build type:
ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug\nPlease call 'make' to build the debug project"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Release\nPlease call 'make' to build the release project"
  )

DEAL_II_INITIALIZE_CACHED_VARIABLES()

DEAL_II_SETUP_TARGET(data_locality)

IF(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
  SET(DEAL_II_HAVE_TESTS_DIRECTORY TRUE)
ENDIF()

enable_testing()

ADD_SUBDIRECTORY(cmake)

ADD_SUBDIRECTORY(benchmark_basic)
ADD_SUBDIRECTORY(benchmark_matrix_power_kernel)
ADD_SUBDIRECTORY(benchmark_merged)
ADD_SUBDIRECTORY(benchmark_precond)
ADD_SUBDIRECTORY(benchmark_precond_comm)
ADD_SUBDIRECTORY(benchmark_precond_merged)
ADD_SUBDIRECTORY(benchmark_precond_pre_vmult_post)
ADD_SUBDIRECTORY(benchmark_range)
ADD_SUBDIRECTORY(benchmark_s_step)
ADD_SUBDIRECTORY(benchmark_pipelined)
ADD_SUBDIRECTORY(benchmark_pipelined_merged)
