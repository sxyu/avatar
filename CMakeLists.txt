cmake_minimum_required( VERSION 2.8 )

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project( smplsynth )
option( WITH_PCL "Enable PCL" OFF )
option( WITH_K4A "Enable K4A (Azure Kinect); needed for demo" ON )
option( WITH_OMP "Enable OMP" OFF )
option( BUILD_RTREE_TOOLS "Build random forest tools" ON )
option( OPENARK_FFAST_MATH "Enable ffast-math compiler flag, may cause numerical problems" ON )

set( INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include" )

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
set( LIB_NAME "libsmplsynth" )
set( SYNTH_NAME "smplsynth" )
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

if ( CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang") )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-deprecated-declarations -O3 -funroll-loops -g" )
    if( ${OPENARK_FFAST_MATH} )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math" )
    endif()
elseif( MSVC )
    if( ${OPENARK_FFAST_MATH} )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast" )
    endif()
endif ( )

if( NOT DEFINED EIGEN_INCLUDE_DIRS )
    if ( DEFINED Eigen_DIR )
        set( EIGEN_INCLUDE_DIRS ${Eigen_DIR} )
    endif ( DEFINED Eigen_DIR )
endif( NOT DEFINED EIGEN_INCLUDE_DIRS )
if ( NOT DEFINED Eigen_DIR )
    find_package( Eigen QUIET )
endif ( NOT DEFINED Eigen_DIR )
set( EIGEN_INCLUDE_DIR ${EIGEN_INCLUDE_DIRS} )
include_directories(${EIGEN_INCLUDE_DIRS})
message(STATUS "Found Eigen: ${EIGEN_INCLUDE_DIRS}")

# require K4A?
set( k4a_FOUND OFF )
if ( ${WITH_K4A} )
    find_package( k4a )
    if( k4a_FOUND)
      message( STATUS "Found k4a (Azure Kinect SDK)" )
    else()
      message( STATUS "k4a (Azure Kinect SDK) not found, live demo, etc disabled" )
    endif( k4a_FOUND )
    set( _FOUND_CAMERA_LIB ON )
else()
    message( STATUS "K4A (Azure Kinect SDK) disabled, configure with -DWITH_K4A=ON to enable" )
endif ()

# require OpenCV
find_package( OpenCV REQUIRED )
if( OpenCV_FOUND )
   message( STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS}" )
endif( OpenCV_FOUND )

set( PCL_FOUND OFF )
if ( ${WITH_PCL} )
    # Find PCL
    find_package( PCL )
    if( PCL_FOUND )
        message( STATUS "Found PCL: ${PCL_INCLUDE_DIRS}" )
    else()
        message( WARNING "PCL not found, disabling AvatarPCL functions" )
    endif( PCL_FOUND )
else()
    message( STATUS "PCL disabled, configure with -DWITH_PCL=ON to enable" )
endif ()

# require Boost
set( Boost_USE_STATIC_LIBS ON )
set( Boost_USE_STATIC ON )
find_package( Boost REQUIRED COMPONENTS filesystem program_options thread system )

# require Ceres
find_package( Ceres REQUIRED )
IF(Ceres_FOUND)
    MESSAGE(STATUS "Found Ceres: ${CERES_INCLUDE_DIRS} ${CERES_LIBRARIES}")
ENDIF(Ceres_FOUND)

if ( ${WITH_OMP} )
    find_package( OpenMP )
    if (OPENMP_FOUND)
      message(STATUS "OpenMP found")
      add_definitions(-DUSE_OPENMP)
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

include_directories(
    ${INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)
 
set(
  DEPENDENCIES
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  ${Boost_LIBRARIES}
)

set(
  SOURCES
  Avatar.cpp 
  BGSubtractor.cpp 
  DepthCamera.cpp 
  RTree.cpp 
  AvatarOptimizer.cpp
  Calibration.cpp 
  GaussianMixture.cpp 
  SparseImage.cpp
  Util.cpp 
  ViconSkeleton.cpp 
)

set(
  HEADERS
  ${INCLUDE_DIR}/Avatar.h
  ${INCLUDE_DIR}/AvatarRenderer.h
  ${INCLUDE_DIR}/AvatarOptimizer.h
  ${INCLUDE_DIR}/BGSubtractor.h
  ${INCLUDE_DIR}/Calibration.h
  ${INCLUDE_DIR}/Config.h
  ${INCLUDE_DIR}/DepthCamera.h
  ${INCLUDE_DIR}/GaussianMixture.h
  ${INCLUDE_DIR}/RTree.h
  ${INCLUDE_DIR}/SparseImage.h
  ${INCLUDE_DIR}/Util.h
  ${INCLUDE_DIR}/ViconSkeleton.h
  ${INCLUDE_DIR}/nanoflann.hpp
)

#  Set up configs
if( NOT k4a_FOUND )
  set( _AZURE_KINECT_SDK_ "//" )
else()
  set( SOURCES ${SOURCES} AzureKinectCamera.cpp )
  set( HEADERS ${HEADERS} ${INCLUDE_DIR}/AzureKinectCamera.h )
  set( DEPENDENCIES ${DEPENDENCIES} k4a::k4a )
endif( NOT k4a_FOUND )

if ( NOT PCL_FOUND )
    set( _WITH_PCL_ "//" )
else()
    set( SOURCES ${SOURCES} AvatarPCL.cpp )
    set( HEADERS ${HEADERS} ${INCLUDE_DIR}/AvatarPCL.h )

    add_definitions(
        ${PCL_DEFINITIONS}
        )
    foreach( DEFINITION ${PCL_DEFINITIONS} )
        set( TARGET_COMPILE_FLAGS "${TARGET_COMPILE_FLAGS} ${DEFINITION}" )
    endforeach()
endif ()

configure_file( Version.h.in ${INCLUDE_DIR}/Version.h )
set( HEADERS ${HEADERS} ${INCLUDE_DIR}/Version.h )

add_library( ${LIB_NAME} STATIC ${SOURCES} ${HEADERS} )
set_target_properties( ${LIB_NAME} PROPERTIES OUTPUT_NAME ${OUTPUT_NAME} )
target_link_libraries( ${LIB_NAME} ${DEPENDENCIES} )
if ( PCL_FOUND )
    set_target_properties( ${LIB_NAME} PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
endif()

add_executable( ${SYNTH_NAME} smplsynth.cpp )
target_include_directories( ${SYNTH_NAME} PRIVATE ${INCLUDE_DIR} )
set_target_properties( ${SYNTH_NAME} PROPERTIES OUTPUT_NAME ${OUTPUT_NAME} )
target_link_libraries( ${SYNTH_NAME} ${DEPENDENCIES} ${LIB_NAME} )
if ( PCL_FOUND )
    set_target_properties( ${SYNTH_NAME} PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
endif()

if( PCL_FOUND )
    add_executable( scratch scratch.cpp )
    target_include_directories( scratch PRIVATE ${INCLUDE_DIR} )
    target_link_libraries( scratch ${DEPENDENCIES} ${LIB_NAME} )
    set_target_properties( scratch PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
endif()

add_executable( bgsubtract bgsubtract.cpp )
target_include_directories( bgsubtract PRIVATE ${INCLUDE_DIR} )
target_link_libraries( bgsubtract ${DEPENDENCIES} ${LIB_NAME} )
if ( PCL_FOUND )
    set_target_properties( bgsubtract PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
endif()

add_executable( smpltrim smpltrim.cpp )
target_include_directories( smpltrim PRIVATE ${INCLUDE_DIR} )
target_link_libraries( smpltrim ${DEPENDENCIES} ${LIB_NAME} )
if ( PCL_FOUND )
    set_target_properties( smpltrim PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
endif()

# RTree stuff
if ( ${BUILD_RTREE_TOOLS} )
    add_executable( rtree-train rtree-train.cpp )
    target_include_directories( rtree-train PRIVATE ${INCLUDE_DIR} )
    target_link_libraries( rtree-train ${DEPENDENCIES} ${LIB_NAME} )
    if ( PCL_FOUND )
        set_target_properties( rtree-train PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
    endif()

    add_executable( rtree-transfer rtree-transfer.cpp )
    target_include_directories( rtree-transfer PRIVATE ${INCLUDE_DIR} )
    target_link_libraries( rtree-transfer ${DEPENDENCIES} ${LIB_NAME} )
    if ( PCL_FOUND )
        set_target_properties( rtree-transfer PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
    endif()

    add_executable( rtree-run rtree-run.cpp )
    target_include_directories( rtree-run PRIVATE ${INCLUDE_DIR} )
    target_link_libraries( rtree-run ${DEPENDENCIES} ${LIB_NAME} )
    if ( PCL_FOUND )
        set_target_properties( rtree-run PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
    endif()

    add_executable( rtree-run-dataset rtree-run-dataset.cpp )
    target_include_directories( rtree-run-dataset PRIVATE ${INCLUDE_DIR} )
    target_link_libraries( rtree-run-dataset ${DEPENDENCIES} ${LIB_NAME} )
    if ( PCL_FOUND )
        set_target_properties( rtree-run-dataset PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
    endif()
endif ()

if ( k4a_FOUND )
    add_executable( live-demo live-demo.cpp )
    target_include_directories( live-demo PRIVATE ${INCLUDE_DIR} )
    target_link_libraries( live-demo ${DEPENDENCIES} ${LIB_NAME} )
    if ( PCL_FOUND )
        set_target_properties( live-demo PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
    endif()
endif ()

if( ${WITH_K4A} AND k4a_FOUND )
    add_executable( data-recording DataRecording.cpp )
    target_include_directories( data-recording PRIVATE ${INCLUDE_DIR} )
    target_link_libraries( data-recording ${DEPENDENCIES} ${LIB_NAME} )
    if ( PCL_FOUND )
        set_target_properties( data-recording PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
    endif()
endif ()

if ( MSVC )
    set_property(TARGET ${PROJ_NAME} APPEND PROPERTY LINK_FLAGS /DEBUG)
else ()
    target_link_libraries( ${PROJ_NAME} -pthread )
endif ( MSVC )
