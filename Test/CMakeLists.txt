# CMakeLists.txt for GTest framework
include(FetchContent)

set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.16.0
)

FetchContent_MakeAvailable(googletest)

add_executable(tensor_tests
    tensor_core_test.cpp
    arithmetic_test.cpp
    autograd_test.cpp
    arena_test.cpp
    nn_test.cpp
)

target_link_libraries(tensor_tests
    PRIVATE
        engine
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tensor_tests)