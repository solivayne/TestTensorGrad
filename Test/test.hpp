// Shared test fixture: resets globalArena before and after each test.
#pragma once

#include <gtest/gtest.h>
#include "Arena.hpp"

class EngineTest : public ::testing::Test {
protected:
    void SetUp() override { globalArena.reset(); }
    void TearDown() override { globalArena.reset(); }
};
