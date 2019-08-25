cmake_minimum_required( VERSION 2.8 )

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project( smplsynth )

set( CMAKE_CXX_STACK_SIZE "10000000" )
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules" )
if (WIN32)
    set( COLOR_TERM OFF )
else()
    set( COLOR_TERM ON )
endif()

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release )
endif()

set( PROJ_NAME "smplsynth" )
set( PART_LABELS_NAME "partlabels" )
set( OUTPUT_NAME "smplsynth" )

include( CheckCXXCompilerFlag )
CHECK_CXX_COMPILER_FLAG( "-std=c++11" COMPILER_SUPPORTS_CXX11 )
CHECK_CXX_COMPILER_FLAG( "-std=c++0x" COMPILER_SUPPORTS_CXX0X )

if( COMPILER_SUPPORTS_CXX11 )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
elseif( COMPILER_SUPPORTS_CXX0X )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x" )
else()
    message( STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler." )
endif()

if ( CMAKE_COMPILER_IS_GNUCXX )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-deprecated-declarations -O3 -g" )
endif ( CMAKE_COMPILER_IS_GNUCXX )

set( Boost_USE_STATIC_LIBS ON )
set( Boost_USE_STATIC ON )

if( NOT DEFINED EIGEN_INCLUDE_DIRS )
    if ( DEFINED Eigen_DIR )
        set( EIGEN_INCLUDE_DIRS ${Eigen_DIR} )
    endif ( DEFINED Eigen_DIR )
endif( NOT DEFINED EIGEN_INCLUDE_DIRS )
if ( NOT DEFINED Eigen_DIR )
    find_package( Eigen QUIET )
endif ( NOT DEFINED Eigen_DIR )
set( EIGEN_INCLUDE_DIR EIGEN_INCLUDE_DIRS )
include_directories(${EIGEN_INCLUDE_DIRS})
message(STATUS "Found Eigen: ${EIGEN_INCLUDE_DIRS}")

# require OpenCV
find_package( OpenCV REQUIRED )
if( OpenCV_FOUND )
   message( STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS}" )
endif( OpenCV_FOUND )

# require PCL
find_package( PCL REQUIRED )
if( PCL_FOUND )
   message( STATUS "Found PCL: ${PCL_INCLUDE_DIRS}" )
endif( PCL_FOUND )

# require OpenARK
find_package( OpenARK REQUIRED )
if( OpenARK_FOUND )
    message( STATUS "Found OpenARK: ${OpenARK_INCLUDE_DIR}" )
endif( OpenARK_FOUND )

# require K4A
find_package( k4a REQUIRED )
if( k4a_FOUND)
  message( STATUS "Found k4a (Azure Kinect SDK)" )
endif( k4a_FOUND )

find_package( Boost REQUIRED COMPONENTS system program_options )

# require Ceres
find_package( Ceres REQUIRED )
IF(Ceres_FOUND)
    MESSAGE(STATUS "Found Ceres: ${CERES_INCLUDE_DIRS} ${CERES_LIBRARIES}")
ENDIF(Ceres_FOUND)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${OpenARK_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)
 
set(
  DEPENDENCIES
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenARK_LIB}
  ${CERES_LIBRARIES}
  ${Boost_LIBRARIES}
  k4a::k4a
)

add_definitions(
  ${PCL_DEFINITIONS}
)
foreach( DEFINITION ${PCL_DEFINITIONS} )
  set( TARGET_COMPILE_FLAGS "${TARGET_COMPILE_FLAGS} ${DEFINITION}" )
endforeach()

add_executable( ${PROJ_NAME} smplsynth.cpp )
target_include_directories( ${PROJ_NAME} PRIVATE ${INCLUDE_DIR} )
set_target_properties( ${PROJ_NAME} PROPERTIES OUTPUT_NAME ${OUTPUT_NAME} )
target_link_libraries( ${PROJ_NAME} ${DEPENDENCIES} )
set_target_properties( ${PROJ_NAME} PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )

add_executable( ${PART_LABELS_NAME} partlabels.cpp )
target_include_directories( ${PART_LABELS_NAME} PRIVATE ${INCLUDE_DIR} )
set_target_properties( ${PART_LABELS_NAME} PROPERTIES OUTPUT_NAME ${PART_LABELS_NAME} )
target_link_libraries( ${PART_LABELS_NAME} ${DEPENDENCIES} )
set_target_properties( ${PART_LABELS_NAME} PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
  
if ( MSVC )
    set_property(TARGET ${PROJ_NAME} APPEND PROPERTY LINK_FLAGS /DEBUG)
else ()
    target_link_libraries( ${PROJ_NAME} -pthread )
endif ( MSVC )
