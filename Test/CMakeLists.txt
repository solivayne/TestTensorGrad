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

add_executable(tests
    # test_tensorcore.cpp
    test_arithmetic.cpp
    # test_autograd.cpp
    # test_arena.cpp
    # test_nn.cpp
)

target_link_libraries(tests
    PRIVATE
        engine
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tests)