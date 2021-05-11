cmake_minimum_required( VERSION 3.3 )

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project( smplsynth )
option( WITH_PCL "Enable PCL" OFF )
option( WITH_K4A "Enable K4A (Azure Kinect); either this or FREENECT2 needed for demo" ON )
option( WITH_FREENECT2 "Enable Freenect2 (Kinect v2); either this or K4A needed for demo" ON )
option( WITH_OMP "Enable OMP" OFF )
option( BUILD_RTREE_TOOLS "Build random forest tools" OFF )

set( INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include" )

set( CMAKE_CXX_STACK_SIZE "10000000" )
set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_STANDARD_REQUIRED ON)
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
set( LIB_NAME "avatar" )
set( SYNTH_NAME "smplsynth" )
set( VENDOR_DIR "${PROJECT_SOURCE_DIR}/3rdparty" )
set( MESHVIEW_DIR "${VENDOR_DIR}/meshview" )

if ( CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang") )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-deprecated-declarations -O3 -funroll-loops -g" )
elseif( MSVC )
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT /GLT /Ox")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MTd")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT /GLT")
    # Silence deprecation warning coming from Eigen; silence secure warning
    add_definitions(-D_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING -D_CRT_SECURE_NO_WARNINGS)
endif ( )

if ( CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang") )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-deprecated-declarations -O3 -funroll-loops -g" )
endif ( )

# Git submodule auto update
# https://cliutils.gitlab.io/modern-cmake/chapters/projects/submodule.html
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Running update --init --recursive")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
        message(STATUS "Submodule update done")
    endif()
endif()

if(NOT EXISTS "${MESHVIEW_DIR}/CMakeLists.txt")
   message(FATAL_ERROR "A submodule as not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

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

set( freenect2_FOUND OFF )
if ( ${WITH_FREENECT2} )
    find_package( freenect2 )
    if( freenect2_FOUND )
      message( STATUS "Found libfreenect2" )
    else()
      message( STATUS "libfreenect2 not found" )
    endif( freenect2_FOUND )
    set( _FOUND_CAMERA_LIB ON )
else()
    message( STATUS "Freenect2 disabled, configure with -DWITH_FREENECT2=ON to enable" )
endif ()

# Meshview
SET(MESHVIEW_BUILD_EXAMPLE OFF CACHE BOOL "meshview example" FORCE)
add_subdirectory(${MESHVIEW_DIR})

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
if (NOT MSVC)
set( Boost_USE_STATIC_LIBS ON )
set( Boost_USE_STATIC ON )
endif()
find_package( Boost REQUIRED COMPONENTS filesystem program_options thread system )
add_definitions( -DBOOST_ALL_NO_LIB )
add_definitions( -DBOOST_ALL_DYN_LINK )

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
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

set(
  DEPENDENCIES
  ${CMAKE_THREAD_LIBS_INIT}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  ${Boost_LIBRARIES}
  ZLIB::ZLIB
  Eigen3::Eigen
)

set(
  SOURCES
  Avatar.cpp
  AvatarModel.cpp
  AvatarRenderer.cpp
  AvatarPoseSequence.cpp
  AvatarHelpers.cpp

  BGSubtractor.cpp
  DepthCamera.cpp
  RTree.cpp
  AvatarOptimizer.cpp
  Calibration.cpp
  GaussianMixture.cpp
  SparseImage.cpp
  Util.cpp
  # ViconSkeleton.cpp
  cnpy.cpp
)

set(
  HEADERS
  ${INCLUDE_DIR}/Avatar.h
  ${INCLUDE_DIR}/AvatarRenderer.h
  ${INCLUDE_DIR}/AvatarOptimizer.h
  ${INCLUDE_DIR}/BGSubtractor.h
  ${INCLUDE_DIR}/Calibration.h
  ${INCLUDE_DIR}/DepthCamera.h
  ${INCLUDE_DIR}/GaussianMixture.h
  ${INCLUDE_DIR}/RTree.h
  ${INCLUDE_DIR}/SparseImage.h
  ${INCLUDE_DIR}/Util.h
  ${INCLUDE_DIR}/UtilCnpy.h
  # ${INCLUDE_DIR}/ViconSkeleton.h
  ${INCLUDE_DIR}/nanoflann.hpp
  ${INCLUDE_DIR}/cnpy.h
)
set(
  PRIVATE_HEADERS
  ${INCLUDE_DIR}/internal/AvatarHelpers.h
)

#  Set up configs
set (FOUND_CAMERA_LIB FALSE)
if( NOT k4a_FOUND )
  set( _AZURE_KINECT_SDK_ "//" )
else()
  set( _AZURE_KINECT_SDK_ "" )
  set( SOURCES ${SOURCES} AzureKinectCamera.cpp )
  set( HEADERS ${HEADERS} ${INCLUDE_DIR}/AzureKinectCamera.h )
  list(APPEND DEPENDENCIES k4a::k4a)
  if (NOT FOUND_CAMERA_LIB)
      set(FOUND_CAMERA_LIB TRUE)
      set( _PREFERRED_CAMERA_ "AzureKinectCamera" )
  endif()
endif( NOT k4a_FOUND )

if( NOT freenect2_FOUND )
  set( _FREENECT2_ "//" )
else()
  set( _FREENECT2_ "" )
  set( SOURCES ${SOURCES} Freenect2Camera.cpp )
  set( HEADERS ${HEADERS} ${INCLUDE_DIR}/Freenect2Camera.h )
  list(APPEND DEPENDENCIES freenect2)
  if (NOT FOUND_CAMERA_LIB)
      set(FOUND_CAMERA_LIB TRUE)
      set( _PREFERRED_CAMERA_ "Freenect2Camera" )
  endif()
endif( NOT freenect2_FOUND )

if (NOT FOUND_CAMERA_LIB)
    message (WARNING No camera lib found, live demo and data recoding will not be built)
endif()

if ( NOT PCL_FOUND )
    set( _WITH_PCL_ "//" )
else()
    set( _WITH_PCL_ "" )
    set( SOURCES ${SOURCES} AvatarPCL.cpp )
    set( HEADERS ${HEADERS} ${INCLUDE_DIR}/AvatarPCL.h )

    add_definitions(
        ${PCL_DEFINITIONS}
        )
    foreach( DEFINITION ${PCL_DEFINITIONS} )
        set( TARGET_COMPILE_FLAGS "${TARGET_COMPILE_FLAGS} ${DEFINITION}" )
    endforeach()
endif ()

configure_file( Version.h.in "${PROJECT_BINARY_DIR}/include/Version.h" )
set( HEADERS ${HEADERS} "${PROJECT_BINARY_DIR}/include/Version.h" )

add_library( ${LIB_NAME} STATIC ${SOURCES} ${HEADERS} )
target_link_libraries( ${LIB_NAME} ${DEPENDENCIES} )
if ( PCL_FOUND )
    set_target_properties( ${LIB_NAME} PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
endif()

add_executable( smpl-viewer smpl-viewer.cpp )
target_link_libraries( smpl-viewer ${DEPENDENCIES} ${LIB_NAME} meshview )

add_executable( ${SYNTH_NAME} smplsynth.cpp )
target_link_libraries( ${SYNTH_NAME} ${DEPENDENCIES} ${LIB_NAME} )
if ( PCL_FOUND )
    set_target_properties( ${SYNTH_NAME} PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
endif()

if( PCL_FOUND )
    add_executable( scratch scratch.cpp )
    target_link_libraries( scratch ${DEPENDENCIES} ${LIB_NAME} )
    set_target_properties( scratch PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
endif()

add_executable( demo demo.cpp )
target_link_libraries( demo ${DEPENDENCIES} ${LIB_NAME} )
if ( PCL_FOUND )
    set_target_properties( demo PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
endif()

# add_executable( smpltrim smpltrim.cpp )
# target_link_libraries( smpltrim ${DEPENDENCIES} ${LIB_NAME} )
# if ( PCL_FOUND )
#     set_target_properties( smpltrim PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
# endif()

# RTree stuff
if ( ${BUILD_RTREE_TOOLS} )
    add_executable( rtree-train rtree-train.cpp )
    target_link_libraries( rtree-train ${DEPENDENCIES} ${LIB_NAME} )
    if ( PCL_FOUND )
        set_target_properties( rtree-train PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
    endif()

    add_executable( rtree-transfer rtree-transfer.cpp )
    target_link_libraries( rtree-transfer ${DEPENDENCIES} ${LIB_NAME} )
    if ( PCL_FOUND )
        set_target_properties( rtree-transfer PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
    endif()

    add_executable( rtree-run rtree-run.cpp )
    target_link_libraries( rtree-run ${DEPENDENCIES} ${LIB_NAME} )
    if ( PCL_FOUND )
        set_target_properties( rtree-run PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
    endif()

    add_executable( rtree-run-dataset rtree-run-dataset.cpp )
    target_link_libraries( rtree-run-dataset ${DEPENDENCIES} ${LIB_NAME} )
    if ( PCL_FOUND )
        set_target_properties( rtree-run-dataset PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
    endif()
endif ()

if ( FOUND_CAMERA_LIB )
    add_executable( live-demo live-demo.cpp )
    target_link_libraries( live-demo ${DEPENDENCIES} ${LIB_NAME} )
    if ( PCL_FOUND )
        set_target_properties( live-demo PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
    endif()
endif ( FOUND_CAMERA_LIB )

if( ${WITH_K4A} AND FOUND_CAMERA_LIB  )
    add_executable( data-recording data-recording.cpp )
    target_link_libraries( data-recording ${DEPENDENCIES} ${LIB_NAME} )
    if ( PCL_FOUND )
        set_target_properties( data-recording PROPERTIES COMPILE_FLAGS ${TARGET_COMPILE_FLAGS} )
    endif()
endif ()

if ( MSVC )
    set_property(TARGET ${PROJ_NAME} APPEND PROPERTY LINK_FLAGS /DEBUG)

endif ( MSVC )
if(WIN32)
	add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
endif()
source_group( "Header Files" FILES ${HEADERS} ${PRIVATE_HEADERS} )
source_group( "Source Files" FILES ${SOURCES} )
