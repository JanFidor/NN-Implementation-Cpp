cmake_minimum_required(VERSION 3.11.0)
project(my_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# include and configure catch2
include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY      https://github.com/catchorg/Catch2.git
    GIT_TAG             v3.0.0-preview3
)
FetchContent_MakeAvailable(Catch2)

# macro to find subdirectories
# https://stackoverflow.com/a/7788165
MACRO(SUBDIRLIST result curdir)
    FILE(GLOB children CONFIGURE_DEPENDS ${curdir}/*)
    SET(dirlist "")
    FOREACH(child ${children})
        IF(IS_DIRECTORY ${child})
            LIST(APPEND dirlist ${child})
        ENDIF()
    ENDFOREACH()
    SET(${result} ${dirlist})
ENDMACRO()

# find subdirectories in src that contain code for classes
SUBDIRLIST(SOURCE_DIRECTORIES src)

# in each subdirectory, find cpp files and add them to the source list
set(SOURCES "")
set(TEST_SOURCES "")
FOREACH(subdir ${SOURCE_DIRECTORIES})
    FILE(GLOB sources CONFIGURE_DEPENDS ${subdir}/*.cpp)
    FILE(GLOB test_sources CONFIGURE_DEPENDS ${subdir}/test*.cpp)

    LIST(APPEND SOURCES ${sources})
    IF(test_sources)
        LIST(REMOVE_ITEM SOURCES ${test_sources})
    ENDIF()

    LIST(APPEND TEST_SOURCES ${test_sources})
ENDFOREACH()

# include the source directories
include_directories(${SOURCE_DIRECTORIES})

# create a build target for the main project file
add_executable(main src/main.cpp ${SOURCES})

# create a build target for the testing file
add_executable(tests src/tests.cpp ${TEST_SOURCES} ${SOURCES})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

# initialize catch2
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(tests)