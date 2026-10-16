/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cuda-qx/core/extension_point.h"
#include "cuda-qx/core/graph.h"
#include "cuda-qx/core/heterogeneous_map.h"
#include "cuda-qx/core/tensor.h"

#include <algorithm>
#include <tuple>

#include <gtest/gtest.h>

namespace cudaqx::testing {

// Define a new extension point for the framework
class MyExtensionPoint : public cudaqx::extension_point<MyExtensionPoint> {
public:
  virtual std::string parrotBack(const std::string &msg) const = 0;
  virtual ~MyExtensionPoint() = default;
};

} // namespace cudaqx::testing

INSTANTIATE_REGISTRY_NO_ARGS(cudaqx::testing::MyExtensionPoint)

namespace cudaqx::testing {

// Define a concrete realization of that extension point
class RepeatBackOne : public MyExtensionPoint {
public:
  std::string parrotBack(const std::string &msg) const override {
    return msg + " from RepeatBackOne.";
  }

  // Extension must provide a creator function
  CUDAQ_EXTENSION_CREATOR_FUNCTION(MyExtensionPoint, RepeatBackOne)
};

// Extensions must register themselves
CUDAQ_EXT_PT_REGISTER_TYPE(RepeatBackOne)

class RepeatBackTwo : public MyExtensionPoint {
public:
  std::string parrotBack(const std::string &msg) const override {
    return msg + " from RepeatBackTwo.";
  }
  CUDAQ_EXTENSION_CREATOR_FUNCTION(MyExtensionPoint, RepeatBackTwo)
};
CUDAQ_EXT_PT_REGISTER_TYPE(RepeatBackTwo)

} // namespace cudaqx::testing

TEST(CoreTester, checkSimpleExtensionPoint) {

  auto registeredNames = cudaqx::testing::MyExtensionPoint::get_registered();
  EXPECT_EQ(registeredNames.size(), 2);
  EXPECT_TRUE(std::find(registeredNames.begin(), registeredNames.end(),
                        "RepeatBackTwo") != registeredNames.end());
  EXPECT_TRUE(std::find(registeredNames.begin(), registeredNames.end(),
                        "RepeatBackOne") != registeredNames.end());
  EXPECT_TRUE(std::find(registeredNames.begin(), registeredNames.end(),
                        "RepeatBackThree") == registeredNames.end());

  {
    auto var = cudaqx::testing::MyExtensionPoint::get("RepeatBackOne");
    EXPECT_EQ(var->parrotBack("Hello World"),
              "Hello World from RepeatBackOne.");
  }
  {
    auto var = cudaqx::testing::MyExtensionPoint::get("RepeatBackTwo");
    EXPECT_EQ(var->parrotBack("Hello World"),
              "Hello World from RepeatBackTwo.");
  }
}

namespace cudaqx::testing {

class MyExtensionPointWithArgs
    : public cudaqx::extension_point<MyExtensionPointWithArgs, int, double> {
protected:
  int i;
  double d;

public:
  MyExtensionPointWithArgs(int i, double d) : i(i), d(d) {}
  virtual std::tuple<int, double, std::string> parrotBack() const = 0;
  virtual ~MyExtensionPointWithArgs() = default;
};

} // namespace cudaqx::testing

INSTANTIATE_REGISTRY(cudaqx::testing::MyExtensionPointWithArgs, int, double)

namespace cudaqx::testing {

class RepeatBackOneWithArgs : public MyExtensionPointWithArgs {
public:
  using MyExtensionPointWithArgs::MyExtensionPointWithArgs;
  std::tuple<int, double, std::string> parrotBack() const override {
    return std::make_tuple(i, d, "RepeatBackOne");
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      RepeatBackOneWithArgs,
      static std::unique_ptr<MyExtensionPointWithArgs> create(int i, double d) {
        return std::make_unique<RepeatBackOneWithArgs>(i, d);
      })
};

CUDAQ_EXT_PT_REGISTER_TYPE(RepeatBackOneWithArgs)

class RepeatBackTwoWithArgs : public MyExtensionPointWithArgs {
public:
  using MyExtensionPointWithArgs::MyExtensionPointWithArgs;
  std::tuple<int, double, std::string> parrotBack() const override {
    return std::make_tuple(i, d, "RepeatBackTwo");
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      RepeatBackTwoWithArgs,
      static std::unique_ptr<MyExtensionPointWithArgs> create(int i, double d) {
        return std::make_unique<RepeatBackTwoWithArgs>(i, d);
      })
};

CUDAQ_EXT_PT_REGISTER_TYPE(RepeatBackTwoWithArgs)

} // namespace cudaqx::testing

TEST(CoreTester, checkSimpleExtensionPointWithArgs) {

  auto registeredNames =
      cudaqx::testing::MyExtensionPointWithArgs::get_registered();
  EXPECT_EQ(registeredNames.size(), 2);
  EXPECT_TRUE(std::find(registeredNames.begin(), registeredNames.end(),
                        "RepeatBackTwoWithArgs") != registeredNames.end());
  EXPECT_TRUE(std::find(registeredNames.begin(), registeredNames.end(),
                        "RepeatBackOneWithArgs") != registeredNames.end());
  EXPECT_TRUE(std::find(registeredNames.begin(), registeredNames.end(),
                        "RepeatBackThree") == registeredNames.end());

  {
    auto var = cudaqx::testing::MyExtensionPointWithArgs::get(
        "RepeatBackOneWithArgs", 5, 2.2);
    auto [i, d, msg] = var->parrotBack();
    EXPECT_EQ(msg, "RepeatBackOne");
    EXPECT_EQ(i, 5);
    EXPECT_NEAR(d, 2.2, 1e-2);
  }
  {
    auto var = cudaqx::testing::MyExtensionPointWithArgs::get(
        "RepeatBackTwoWithArgs", 15, 12.2);
    auto [i, d, msg] = var->parrotBack();
    EXPECT_EQ(msg, "RepeatBackTwo");
    EXPECT_EQ(i, 15);
    EXPECT_NEAR(d, 12.2, 1e-2);
  }
}

TEST(CoreTester, checkTensorSimple) {
  auto registeredNames = cudaqx::details::tensor_impl<>::get_registered();
  EXPECT_EQ(registeredNames.size(), 1);
  EXPECT_TRUE(std::find(registeredNames.begin(), registeredNames.end(),
                        "xtensorcomplex<double>") != registeredNames.end());

  {
    cudaqx::tensor t({1, 2, 1});
    EXPECT_EQ(t.rank(), 3);
    EXPECT_EQ(t.size(), 2);
    for (std::size_t i = 0; i < 1; i++)
      for (std::size_t j = 0; j < 2; j++)
        for (std::size_t k = 0; k < 1; k++)
          EXPECT_NEAR(t.at({i, j, k}).real(), 0.0, 1e-8);

    t.at({0, 1, 0}) = 2.2;
    EXPECT_NEAR(t.at({0, 1, 0}).real(), 2.2, 1e-8);

    EXPECT_ANY_THROW({ t.at({2, 2, 2}); });
  }

  {
    cudaqx::tensor t({2, 2});
    EXPECT_EQ(t.rank(), 2);
    EXPECT_EQ(t.size(), 4);
    std::vector<std::complex<double>> data{1, 2, 3, 4};
    t.copy(data.data(), {2, 2});
    EXPECT_NEAR(t.at({0, 0}).real(), 1., 1e-8);
    EXPECT_NEAR(t.at({0, 1}).real(), 2., 1e-8);
    EXPECT_NEAR(t.at({1, 0}).real(), 3., 1e-8);
    EXPECT_NEAR(t.at({1, 1}).real(), 4., 1e-8);
  }
  {
    cudaqx::tensor t({2, 2});
    EXPECT_EQ(t.rank(), 2);
    EXPECT_EQ(t.size(), 4);
    std::vector<std::complex<double>> data{1, 2, 3, 4};
    t.copy(data.data());
    EXPECT_NEAR(t.at({0, 0}).real(), 1., 1e-8);
    EXPECT_NEAR(t.at({0, 1}).real(), 2., 1e-8);
    EXPECT_NEAR(t.at({1, 0}).real(), 3., 1e-8);
    EXPECT_NEAR(t.at({1, 1}).real(), 4., 1e-8);
  }
  {
    cudaqx::tensor t({2, 2});
    EXPECT_EQ(t.rank(), 2);
    EXPECT_EQ(t.size(), 4);
    std::vector<std::complex<double>> data{1, 2, 3, 4};
    t.borrow(data.data(), {2, 2});
    EXPECT_NEAR(t.at({0, 0}).real(), 1., 1e-8);
    EXPECT_NEAR(t.at({0, 1}).real(), 2., 1e-8);
    EXPECT_NEAR(t.at({1, 0}).real(), 3., 1e-8);
    EXPECT_NEAR(t.at({1, 1}).real(), 4., 1e-8);
  }

  {
    cudaqx::tensor t({2, 2});
    EXPECT_EQ(t.rank(), 2);
    EXPECT_EQ(t.size(), 4);
    std::vector<std::complex<double>> data{1, 2, 3, 4};
    t.borrow(data.data());
    EXPECT_NEAR(t.at({0, 0}).real(), 1., 1e-8);
    EXPECT_NEAR(t.at({0, 1}).real(), 2., 1e-8);
    EXPECT_NEAR(t.at({1, 0}).real(), 3., 1e-8);
    EXPECT_NEAR(t.at({1, 1}).real(), 4., 1e-8);
  }
  {
    cudaqx::tensor t;
    std::vector<std::complex<double>> data{1, 2, 3, 4};
    EXPECT_THROW({ t.borrow(data.data()); }, std::runtime_error);
  }
  {
    cudaqx::tensor t;
    std::vector<std::complex<double>> data{1, 2, 3, 4};
    EXPECT_THROW({ t.copy(data.data()); }, std::runtime_error);
  }
  {
    cudaqx::tensor t;
    std::vector<std::complex<double>> data{1, 2, 3, 4};
    EXPECT_THROW({ t.take(data.data()); }, std::runtime_error);
  }
  {
    cudaqx::tensor t({2, 2});
    EXPECT_EQ(t.rank(), 2);
    EXPECT_EQ(t.size(), 4);
    std::complex<double> *data = new std::complex<double>[4];
    double count = 1.0;
    std::generate_n(data, 4, [&]() { return count++; });
    t.take(data, {2, 2});
    EXPECT_NEAR(t.at({0, 0}).real(), 1., 1e-8);
    EXPECT_NEAR(t.at({0, 1}).real(), 2., 1e-8);
    EXPECT_NEAR(t.at({1, 0}).real(), 3., 1e-8);
    EXPECT_NEAR(t.at({1, 1}).real(), 4., 1e-8);
  }

  {
    cudaqx::tensor<int> t({1, 2, 1});
    EXPECT_EQ(t.rank(), 3);
    EXPECT_EQ(t.size(), 2);
    for (std::size_t i = 0; i < 1; i++)
      for (std::size_t j = 0; j < 2; j++)
        for (std::size_t k = 0; k < 1; k++)
          EXPECT_NEAR(t.at({i, j, k}), 0.0, 1e-8);

    t.at({0, 1, 0}) = 2;
    EXPECT_EQ(t.at({0, 1, 0}), 2);

    EXPECT_ANY_THROW({ t.at({2, 2, 2}); });
  }

  {
    const cudaqx::tensor<int> t({1, 2, 1});
    EXPECT_NEAR(t.at({0, 0, 0}), 0.0, 1e-8);
    EXPECT_THROW(t.at({0, 1}), std::runtime_error);
  }

  {
    cudaqx::tensor<double> a({2, 3});
    cudaqx::tensor<double> v({3});
    EXPECT_EQ(a.rank(), 2);
    EXPECT_EQ(v.rank(), 1);
    testing::internal::CaptureStdout();
    a.dump_bits();
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output, "...\n...\n");

    testing::internal::CaptureStdout();
    v.dump_bits();
    output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output, "...\n");
  }
}

TEST(TensorTest, checkBitStringConstruction) {
  std::vector<std::string> bitstrings;
  cudaqx::tensor<uint8_t> t_empty(bitstrings);
  EXPECT_EQ(t_empty.rank(), 0);

  bitstrings.push_back("000"); // Shot 0
  bitstrings.push_back("001"); // Shot 1

  cudaqx::tensor<uint8_t> t(bitstrings);
  EXPECT_EQ(t.at({0, 0}), 0);
  EXPECT_EQ(t.at({0, 1}), 0);
  EXPECT_EQ(t.at({0, 2}), 0);
  EXPECT_EQ(t.at({1, 0}), 0);
  EXPECT_EQ(t.at({1, 1}), 0);
  EXPECT_EQ(t.at({1, 2}), 1);
}

// Test elementwise operations
TEST(TensorTest, ElementwiseAddition) {
  cudaqx::tensor<double> a({2, 2});
  cudaqx::tensor<double> b({2, 2});

  // Initialize test data
  double data_a[] = {1.0, 2.0, 3.0, 4.0};
  double data_b[] = {5.0, 6.0, 7.0, 8.0};
  a.copy(data_a);
  b.copy(data_b);

  auto result = a + b;

  // Check result dimensions
  EXPECT_EQ(result.rank(), 2);
  EXPECT_EQ(result.shape()[0], 2);
  EXPECT_EQ(result.shape()[1], 2);

  // Check elementwise addition results
  EXPECT_DOUBLE_EQ(result.at({0, 0}), 6.0);  // 1 + 5
  EXPECT_DOUBLE_EQ(result.at({0, 1}), 8.0);  // 2 + 6
  EXPECT_DOUBLE_EQ(result.at({1, 0}), 10.0); // 3 + 7
  EXPECT_DOUBLE_EQ(result.at({1, 1}), 12.0); // 4 + 8
}

TEST(TensorTest, ElementwiseMultiplication) {
  cudaqx::tensor<double> a({2, 2});
  cudaqx::tensor<double> b({2, 2});

  double data_a[] = {1.0, 2.0, 3.0, 4.0};
  double data_b[] = {5.0, 6.0, 7.0, 8.0};
  a.copy(data_a);
  b.copy(data_b);

  auto result = a * b;

  EXPECT_EQ(result.rank(), 2);
  EXPECT_EQ(result.shape()[0], 2);
  EXPECT_EQ(result.shape()[1], 2);

  EXPECT_DOUBLE_EQ(result.at({0, 0}), 5.0);  // 1 * 5
  EXPECT_DOUBLE_EQ(result.at({0, 1}), 12.0); // 2 * 6
  EXPECT_DOUBLE_EQ(result.at({1, 0}), 21.0); // 3 * 7
  EXPECT_DOUBLE_EQ(result.at({1, 1}), 32.0); // 4 * 8
}

TEST(TensorTest, ElementwiseModulo) {
  cudaqx::tensor<int> a({2, 2});
  cudaqx::tensor<int> b({2, 2});

  int data_a[] = {7, 8, 9, 10};
  int data_b[] = {4, 3, 5, 2};
  a.copy(data_a);
  b.copy(data_b);

  auto result = a % b;

  EXPECT_EQ(result.rank(), 2);
  EXPECT_EQ(result.shape()[0], 2);
  EXPECT_EQ(result.shape()[1], 2);

  EXPECT_EQ(result.at({0, 0}), 3); // 7 % 4
  EXPECT_EQ(result.at({0, 1}), 2); // 8 % 3
  EXPECT_EQ(result.at({1, 0}), 4); // 9 % 5
  EXPECT_EQ(result.at({1, 1}), 0); // 10 % 2
}

TEST(TensorTest, Any) {
  {
    cudaqx::tensor<uint8_t> a({2, 2});

    uint8_t data_a[] = {7, 8, 9, 10};
    a.copy(data_a);

    uint8_t result = a.any();

    EXPECT_TRUE(result);
  }
  {
    cudaqx::tensor<uint8_t> a({2, 2});

    uint8_t data_a[] = {0, 0, 1, 0};
    a.copy(data_a);

    uint8_t result = a.any();

    EXPECT_TRUE(result);
  }
  {
    cudaqx::tensor<uint8_t> a({2, 2});

    uint8_t data_a[] = {0, 0, 0, 0};
    a.copy(data_a);

    uint8_t result = a.any();

    EXPECT_FALSE(result);
  }
}

TEST(TensorTest, SumAll) {
  {
    // test int
    cudaqx::tensor<int> a({2, 2});

    int data_a[] = {7, 8, 9, 10};
    a.copy(data_a);

    int result = a.sum_all();

    EXPECT_EQ(result, 34);
  }
  {
    // test uint8_t
    cudaqx::tensor<uint8_t> a({2, 2});

    uint8_t data_a[] = {7, 8, 9, 10};
    a.copy(data_a);

    uint8_t result = a.sum_all();

    EXPECT_EQ(result, 34);
  }
  {
    // test uint8_t overflow
    cudaqx::tensor<uint8_t> a({2, 2});

    uint8_t data_a[] = {70, 80, 90, 100};
    a.copy(data_a);

    uint8_t result = a.sum_all();

    EXPECT_NE(result, 340);
    EXPECT_EQ(result, (uint8_t)340);
  }
  {
    // test float
    cudaqx::tensor<float> a({2, 2});

    float data_a[] = {7.1, 8.2, 9.1, 10.3};
    a.copy(data_a);

    float result = a.sum_all();

    float tolerance = 1.e-5;
    EXPECT_FLOAT_EQ(result, 34.7);
  }
}

TEST(TensorTest, ScalarModulo) {
  {
    // test int
    cudaqx::tensor<int> a({2, 2});

    int data_a[] = {7, 8, 9, 10};
    a.copy(data_a);

    auto result = a % 2;

    EXPECT_EQ(result.rank(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 2);

    EXPECT_EQ(result.at({0, 0}), 1); // 7 % 2
    EXPECT_EQ(result.at({0, 1}), 0); // 8 % 2
    EXPECT_EQ(result.at({1, 0}), 1); // 9 % 2
    EXPECT_EQ(result.at({1, 1}), 0); // 10 % 2
  }
  {
    // test uint8_t
    cudaqx::tensor<uint8_t> a({2, 2});

    uint8_t data_a[] = {7, 8, 9, 10};
    a.copy(data_a);

    auto result = a % 2;

    EXPECT_EQ(result.rank(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 2);

    EXPECT_EQ(result.at({0, 0}), 1); // 7 % 2
    EXPECT_EQ(result.at({0, 1}), 0); // 8 % 2
    EXPECT_EQ(result.at({1, 0}), 1); // 9 % 2
    EXPECT_EQ(result.at({1, 1}), 0); // 10 % 2
  }
  {
    // test result tensor is input tensor
    cudaqx::tensor<uint8_t> a({2, 2});

    uint8_t data_a[] = {7, 8, 9, 10};
    a.copy(data_a);

    a = a % 2;

    EXPECT_EQ(a.rank(), 2);
    EXPECT_EQ(a.shape()[0], 2);
    EXPECT_EQ(a.shape()[1], 2);

    EXPECT_EQ(a.at({0, 0}), 1); // 7 % 2
    EXPECT_EQ(a.at({0, 1}), 0); // 8 % 2
    EXPECT_EQ(a.at({1, 0}), 1); // 9 % 2
    EXPECT_EQ(a.at({1, 1}), 0); // 10 % 2
  }
}

TEST(TensorTest, MatrixDotProduct) {
  cudaqx::tensor<double> a({2, 3});
  cudaqx::tensor<double> b({3, 2});

  double data_a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double data_b[] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  a.copy(data_a);
  b.copy(data_b);

  auto result = a.dot(b);

  EXPECT_EQ(result.rank(), 2);
  EXPECT_EQ(result.shape()[0], 2);
  EXPECT_EQ(result.shape()[1], 2);

  // Matrix multiplication results
  EXPECT_DOUBLE_EQ(result.at({0, 0}), 58.0);  // 1*7 + 2*9 + 3*11
  EXPECT_DOUBLE_EQ(result.at({0, 1}), 64.0);  // 1*8 + 2*10 + 3*12
  EXPECT_DOUBLE_EQ(result.at({1, 0}), 139.0); // 4*7 + 5*9 + 6*11
  EXPECT_DOUBLE_EQ(result.at({1, 1}), 154.0); // 4*8 + 5*10 + 6*12
}

TEST(TensorTest, MatrixVectorProduct) {
  cudaqx::tensor<double> a({2, 3});
  cudaqx::tensor<double> v({3});

  double data_a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double data_v[] = {7.0, 8.0, 9.0};
  a.copy(data_a);
  v.copy(data_v);

  auto result = a.dot(v);

  EXPECT_EQ(result.rank(), 1);
  EXPECT_EQ(result.shape()[0], 2);

  EXPECT_DOUBLE_EQ(result.at({0}), 50.0);  // 1*7 + 2*8 + 3*9
  EXPECT_DOUBLE_EQ(result.at({1}), 122.0); // 4*7 + 5*8 + 6*9
}

TEST(TensorTest, MatrixTranspose) {
  cudaqx::tensor<double> a({2, 3});

  double data_a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  a.copy(data_a);

  auto result = a.transpose();

  EXPECT_EQ(a.rank(), 2);
  EXPECT_EQ(a.shape()[0], 2);
  EXPECT_EQ(a.shape()[1], 3);

  EXPECT_EQ(result.rank(), 2);
  EXPECT_EQ(result.shape()[0], 3);
  EXPECT_EQ(result.shape()[1], 2);

  EXPECT_DOUBLE_EQ(a.at({0, 0}), 1.0);
  EXPECT_DOUBLE_EQ(a.at({0, 1}), 2.0);
  EXPECT_DOUBLE_EQ(a.at({0, 2}), 3.0);
  EXPECT_DOUBLE_EQ(a.at({1, 0}), 4.0);
  EXPECT_DOUBLE_EQ(a.at({1, 1}), 5.0);
  EXPECT_DOUBLE_EQ(a.at({1, 2}), 6.0);

  EXPECT_DOUBLE_EQ(result.at({0, 0}), 1.0);
  EXPECT_DOUBLE_EQ(result.at({0, 1}), 4.0);
  EXPECT_DOUBLE_EQ(result.at({1, 0}), 2.0);
  EXPECT_DOUBLE_EQ(result.at({1, 1}), 5.0);
  EXPECT_DOUBLE_EQ(result.at({2, 0}), 3.0);
  EXPECT_DOUBLE_EQ(result.at({2, 1}), 6.0);
}

// Test error conditions
TEST(TensorTest, MismatchedShapeAddition) {
  cudaqx::tensor<double> a({2, 2});
  cudaqx::tensor<double> b({2, 3});

  EXPECT_THROW(a + b, std::runtime_error);
}

TEST(TensorTest, InvalidDotProductDimensions) {
  cudaqx::tensor<double> a({2, 3});
  cudaqx::tensor<double> b({2, 2});

  EXPECT_THROW(a.dot(b), std::runtime_error);
}

TEST(TensorTest, InvalidMatrixVectorDimensions) {
  cudaqx::tensor<double> a({2, 3});
  cudaqx::tensor<double> v({2});

  EXPECT_THROW(a.dot(v), std::runtime_error);
}

TEST(TensorTest, ConstructorWithShape) {
  std::vector<std::size_t> shape = {2, 3, 4};
  cudaqx::tensor t(shape);

  EXPECT_EQ(t.rank(), 3);
  EXPECT_EQ(t.size(), 24);
  EXPECT_EQ(t.shape(), shape);
}

TEST(TensorTest, ConstructorWithDataAndShape) {
  std::vector<std::size_t> shape = {2, 2};
  std::complex<double> *data = new std::complex<double>[4];
  data[0] = {1.0, 0.0};
  data[1] = {0.0, 1.0};
  data[2] = {0.0, -1.0};
  data[3] = {1.0, 0.0};

  cudaqx::tensor t(data, shape);

  EXPECT_EQ(t.rank(), 2);
  EXPECT_EQ(t.size(), 4);
  EXPECT_EQ(t.shape(), shape);

  // Check if data is correctly stored
  EXPECT_EQ(t.at({0, 0}), std::complex<double>(1.0, 0.0));
  EXPECT_EQ(t.at({0, 1}), std::complex<double>(0.0, 1.0));
  EXPECT_EQ(t.at({1, 0}), std::complex<double>(0.0, -1.0));
  EXPECT_EQ(t.at({1, 1}), std::complex<double>(1.0, 0.0));
}

TEST(TensorTest, AccessElements) {
  std::vector<std::size_t> shape = {2, 3};
  cudaqx::tensor t(shape);

  // Set values
  t.at({0, 0}) = {1.0, 0.0};
  t.at({0, 1}) = {0.0, 1.0};
  t.at({1, 2}) = {-1.0, 0.0};

  // Check values
  EXPECT_EQ(t.at({0, 0}), std::complex<double>(1.0, 0.0));
  EXPECT_EQ(t.at({0, 1}), std::complex<double>(0.0, 1.0));
  EXPECT_EQ(t.at({1, 2}), std::complex<double>(-1.0, 0.0));
}

TEST(TensorTest, CopyData) {
  std::vector<std::size_t> shape = {2, 2};
  std::vector<std::complex<double>> data = {
      {1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}, {1.0, 0.0}};
  cudaqx::tensor t(shape);

  t.copy(data.data(), shape);

  EXPECT_EQ(t.at({0, 0}), std::complex<double>(1.0, 0.0));
  EXPECT_EQ(t.at({0, 1}), std::complex<double>(0.0, 1.0));
  EXPECT_EQ(t.at({1, 0}), std::complex<double>(0.0, -1.0));
  EXPECT_EQ(t.at({1, 1}), std::complex<double>(1.0, 0.0));
}

TEST(TensorTest, TakeData) {
  std::vector<std::size_t> shape = {2, 2};
  auto data = new std::complex<double>[4]{
      {1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}, {1.0, 0.0}};
  cudaqx::tensor t(shape);

  t.take(data, shape);

  EXPECT_EQ(t.at({0, 0}), std::complex<double>(1.0, 0.0));
  EXPECT_EQ(t.at({0, 1}), std::complex<double>(0.0, 1.0));
  EXPECT_EQ(t.at({1, 0}), std::complex<double>(0.0, -1.0));
  EXPECT_EQ(t.at({1, 1}), std::complex<double>(1.0, 0.0));

  // Note: We don't delete data here as the tensor now owns it
}

TEST(TensorTest, BorrowData) {
  std::vector<std::size_t> shape = {2, 2};
  std::vector<std::complex<double>> data = {
      {1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}, {1.0, 0.0}};
  cudaqx::tensor t(shape);

  t.borrow(data.data(), shape);

  EXPECT_EQ(t.at({0, 0}), std::complex<double>(1.0, 0.0));
  EXPECT_EQ(t.at({0, 1}), std::complex<double>(0.0, 1.0));
  EXPECT_EQ(t.at({1, 0}), std::complex<double>(0.0, -1.0));
  EXPECT_EQ(t.at({1, 1}), std::complex<double>(1.0, 0.0));
}

TEST(TensorTest, InvalidAccess) {
  std::vector<std::size_t> shape = {2, 2};
  cudaqx::tensor t(shape);

  EXPECT_THROW(t.at({2, 0}), std::runtime_error);
  EXPECT_THROW(t.at({0, 2}), std::runtime_error);
  EXPECT_THROW(t.at({0, 0, 0}), std::runtime_error);
}

TEST(TensorTest, checkNullaryConstructor) {
  std::vector<std::size_t> shape = {2, 2};
  std::vector<std::complex<double>> data = {
      {1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}, {1.0, 0.0}};
  cudaqx::tensor t;

  t.copy(data.data(), shape);
}

TEST(HeterogeneousMapTest, checkSimple) {

  {
    cudaqx::heterogeneous_map m;
    m.insert("hello", 2.2);
    m.insert("another", 1);
    m.insert("string", "string");
    EXPECT_EQ(3, m.size());
    EXPECT_NEAR(2.2, m.get<double>("hello"), 1e-3);
    EXPECT_EQ(1, m.get<int>("another"));
    // If the value is int-like, can get it as other int-like types
    EXPECT_EQ(1, m.get<std::size_t>("another"));
    // same for float/double
    EXPECT_NEAR(2.2, m.get<float>("hello"), 1e-3);
    EXPECT_EQ("string", m.get<std::string>("string"));
    EXPECT_EQ("defaulted", m.get<std::string>("key22", "defaulted"));
  }

  {
    cudaqx::heterogeneous_map m({{"hello", 2.2}, {"string", "stringVal"}});
    EXPECT_EQ(2, m.size());
    EXPECT_NEAR(2.2, m.get<float>("hello"), 1e-3);
    EXPECT_EQ("stringVal", m.get<std::string>("string"));
  }
}

TEST(HeterogeneousMapTest, InsertAndRetrieve) {
  cudaqx::heterogeneous_map map;
  map.insert("int_key", 42);
  map.insert("string_key", std::string("hello"));
  map.insert("double_key", 3.14);

  EXPECT_EQ(map.get<int>("int_key"), 42);
  EXPECT_EQ(map.get<std::string>("string_key"), "hello");
  EXPECT_DOUBLE_EQ(map.get<double>("double_key"), 3.14);
}

TEST(HeterogeneousMapTest, InsertOverwrite) {
  cudaqx::heterogeneous_map map;
  map.insert("key", 10);
  EXPECT_EQ(map.get<int>("key"), 10);

  map.insert("key", 20);
  EXPECT_EQ(map.get<int>("key"), 20);
}

TEST(HeterogeneousMapTest, GetWithDefault) {
  cudaqx::heterogeneous_map map;
  EXPECT_EQ(map.get("nonexistent_key", 100), 100);
  EXPECT_EQ(map.get("nonexistent_key", std::string("default")), "default");
}

TEST(HeterogeneousMapTest, Contains) {
  cudaqx::heterogeneous_map map;
  map.insert("existing_key", 42);

  EXPECT_TRUE(map.contains("existing_key"));
  EXPECT_FALSE(map.contains("nonexistent_key"));
  EXPECT_FALSE(map.contains(
      std::vector<std::string>{"nonexistent_key1", "nonexistent_key2"}));
}

TEST(HeterogeneousMapTest, Size) {
  cudaqx::heterogeneous_map map;
  EXPECT_EQ(map.size(), 0);

  map.insert("key1", 10);
  map.insert("key2", "value");

  EXPECT_EQ(map.size(), 2);
}

TEST(HeterogeneousMapTest, Clear) {
  cudaqx::heterogeneous_map map;
  map.insert("key1", 10);
  map.insert("key2", "value");

  EXPECT_EQ(map.size(), 2);

  map.clear();

  EXPECT_EQ(map.size(), 0);
  EXPECT_FALSE(map.contains("key1"));
  EXPECT_FALSE(map.contains("key2"));
}

TEST(HeterogeneousMapTest, RelatedTypes) {
  cudaqx::heterogeneous_map map;
  map.insert("int_key", 42);

  EXPECT_EQ(map.get<std::size_t>("int_key"), 42);
  EXPECT_EQ(map.get<long>("int_key"), 42);
  EXPECT_EQ(map.get<short>("int_key"), 42);
}

// Python kwargs store int as std::size_t; get<bool> must accept it
// (RelatedTypesMap<bool>).
TEST(HeterogeneousMapTest, GetBoolFromSizeT) {
  cudaqx::heterogeneous_map map;
  map.insert("flag", std::size_t(1));
  map.insert("off", std::size_t(0));

  EXPECT_TRUE(map.get<bool>("flag"));
  EXPECT_FALSE(map.get<bool>("off"));
  EXPECT_TRUE(map.get<bool>("flag", false));
  EXPECT_FALSE(map.get<bool>("off", true));
  EXPECT_FALSE(map.get<bool>("missing", false));
  EXPECT_TRUE(map.get<bool>("missing", true));
}

TEST(HeterogeneousMapTest, CharArrayConversion) {
  cudaqx::heterogeneous_map map;
  const char *cstr = "Hello";
  map.insert("char_array_key", cstr);

  EXPECT_EQ(map.get<std::string>("char_array_key"), "Hello");
}

TEST(HeterogeneousMapTest, ExceptionHandling) {
  cudaqx::heterogeneous_map map;
  map.insert("int_key", 42);

  EXPECT_THROW(map.get<std::string>("int_key"), std::runtime_error);
  EXPECT_THROW(map.get<int>("nonexistent_key"), std::runtime_error);
}

TEST(HeterogeneousMapTest, CopyConstructor) {
  cudaqx::heterogeneous_map map;
  map.insert("key1", 10);
  map.insert("key2", "value");

  cudaqx::heterogeneous_map copy_map(map);

  EXPECT_EQ(copy_map.size(), 2);
  EXPECT_EQ(copy_map.get<int>("key1"), 10);
  EXPECT_EQ(copy_map.get<std::string>("key2"), "value");
}

TEST(HeterogeneousMapTest, AssignmentOperator) {
  cudaqx::heterogeneous_map map;
  map.insert("key1", 10);
  map.insert("key2", "value");

  cudaqx::heterogeneous_map assigned_map;
  assigned_map = map;

  EXPECT_EQ(assigned_map.size(), 2);
  EXPECT_EQ(assigned_map.get<int>("key1"), 10);
  EXPECT_EQ(assigned_map.get<std::string>("key2"), "value");
}

TEST(HeterogeneousMapTest, InitializerListConstructor) {
  cudaqx::heterogeneous_map map{{"int_key", 42},
                                {"string_key", std::string("hello")},
                                {"double_key", 3.14}};

  EXPECT_EQ(map.size(), 3);
  EXPECT_EQ(map.get<int>("int_key"), 42);
  EXPECT_EQ(map.get<std::string>("string_key"), "hello");
  EXPECT_DOUBLE_EQ(map.get<double>("double_key"), 3.14);
}

TEST(HeterogeneousMapTest, MoveConstructor) {
  cudaqx::heterogeneous_map original_map;
  original_map.insert("int_key", 42);
  original_map.insert("string_key", std::string("hello"));
  original_map.insert("double_key", 3.14);

  EXPECT_EQ(original_map.size(), 3);

  // Move construct new map
  cudaqx::heterogeneous_map moved_map(std::move(original_map));

  // Verify the moved map has the data
  EXPECT_EQ(moved_map.size(), 3);
  EXPECT_EQ(moved_map.get<int>("int_key"), 42);
  EXPECT_EQ(moved_map.get<std::string>("string_key"), "hello");
  EXPECT_DOUBLE_EQ(moved_map.get<double>("double_key"), 3.14);

  // Original map should be empty after move
  EXPECT_EQ(original_map.size(), 0);
}

TEST(HeterogeneousMapTest, MoveAssignmentOperator) {
  cudaqx::heterogeneous_map source_map;
  source_map.insert("int_key", 42);
  source_map.insert("string_key", std::string("hello"));
  source_map.insert("double_key", 3.14);

  cudaqx::heterogeneous_map target_map;
  target_map.insert("old_key", std::string("old_value"));

  EXPECT_EQ(source_map.size(), 3);
  EXPECT_EQ(target_map.size(), 1);

  // Move assign
  target_map = std::move(source_map);

  // Verify the target map has the moved data
  EXPECT_EQ(target_map.size(), 3);
  EXPECT_EQ(target_map.get<int>("int_key"), 42);
  EXPECT_EQ(target_map.get<std::string>("string_key"), "hello");
  EXPECT_DOUBLE_EQ(target_map.get<double>("double_key"), 3.14);

  // Old data should be gone
  EXPECT_FALSE(target_map.contains("old_key"));

  // Source map should be empty after move
  EXPECT_EQ(source_map.size(), 0);
}

TEST(HeterogeneousMapTest, InsertMoveVector) {
  cudaqx::heterogeneous_map map;
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  map.insert("vec", std::move(data));

  EXPECT_TRUE(data.empty()); // source was moved from
  auto retrieved = map.get<std::vector<double>>("vec");
  EXPECT_EQ(retrieved.size(), 4u);
  EXPECT_DOUBLE_EQ(retrieved[0], 1.0);
  EXPECT_DOUBLE_EQ(retrieved[3], 4.0);
}

TEST(HeterogeneousMapTest, InsertMoveLargePayload) {
  cudaqx::heterogeneous_map map;
  std::vector<std::vector<float>> llr_history(100,
                                              std::vector<float>(200, 0.5f));
  map.insert("llr_history", std::move(llr_history));

  EXPECT_TRUE(llr_history.empty()); // outer vector was moved from
  auto stored = map.get<std::vector<std::vector<float>>>("llr_history");
  EXPECT_EQ(stored.size(), 100u);
  EXPECT_EQ(stored[0].size(), 200u);
  EXPECT_FLOAT_EQ(stored[0][0], 0.5f);
}

TEST(HeterogeneousMapTest, InsertMoveOverwrites) {
  cudaqx::heterogeneous_map map;
  map.insert("key", std::vector<int>{1, 2, 3});
  map.insert("key", std::vector<int>{10, 20});

  auto v = map.get<std::vector<int>>("key");
  EXPECT_EQ(v.size(), 2u);
  EXPECT_EQ(v[0], 10);
  EXPECT_EQ(v[1], 20);
}

TEST(HeterogeneousMapTest, InsertLvalueNotMoved) {
  cudaqx::heterogeneous_map map;
  std::vector<int> src = {7, 8, 9};
  map.insert("vec", src); // lvalue — must go through copy overload

  EXPECT_EQ(src.size(), 3u); // source intact
  EXPECT_EQ(map.get<std::vector<int>>("vec"), src);
}

TEST(GraphTester, AddEdge) {
  cudaqx::graph g;
  g.add_edge(1, 2, 1.5);
  EXPECT_EQ(g.get_neighbors(1), std::vector<int>{2});
  EXPECT_EQ(g.get_neighbors(2), std::vector<int>{1});
  std::vector<std::pair<int, double>> tmp{{2, 1.5}}, tmp2{{1, 1.5}};
  EXPECT_EQ(g.get_weighted_neighbors(1), tmp);
  EXPECT_EQ(g.get_weighted_neighbors(2), tmp2);
}

TEST(GraphTester, AddEdgeDefaultWeight) {
  cudaqx::graph g;
  g.add_edge(1, 2); // Default weight should be 1.0
  EXPECT_EQ(g.get_neighbors(1), std::vector<int>{2});
  EXPECT_EQ(g.get_neighbors(2), std::vector<int>{1});
  std::vector<std::pair<int, double>> tmp{{2, 1.0}};
  EXPECT_EQ(g.get_weighted_neighbors(1), tmp);
}

TEST(GraphTester, AddNode) {
  cudaqx::graph g;
  g.add_node(1);
  EXPECT_EQ(g.get_nodes(), std::vector<int>{1});
}

TEST(GraphTester, GetNeighbors) {
  cudaqx::graph g;
  g.add_edge(1, 2, 0.5);
  g.add_edge(1, 3, 1.5);
  g.add_edge(2, 3, 2.0);
  std::vector<int> tmp{2, 3}, tmp2{1, 2}, tmp3{1, 3}, tmp4{};

  EXPECT_EQ(g.get_neighbors(1), tmp);
  EXPECT_EQ(g.get_neighbors(2), tmp3);
  EXPECT_EQ(g.get_neighbors(3), tmp2);
  EXPECT_EQ(g.get_neighbors(4), tmp4);
}

TEST(GraphTester, GetWeightedNeighbors) {
  cudaqx::graph g;
  g.add_edge(1, 2, 0.5);
  g.add_edge(1, 3, 1.5);
  g.add_edge(2, 3, 2.0);

  std::vector<std::pair<int, double>> expected1 = {{2, 0.5}, {3, 1.5}};
  std::vector<std::pair<int, double>> expected2 = {{1, 0.5}, {3, 2.0}};
  std::vector<std::pair<int, double>> expected3 = {{1, 1.5}, {2, 2.0}};
  std::vector<std::pair<int, double>> expected4 = {};

  EXPECT_EQ(g.get_weighted_neighbors(1), expected1);
  EXPECT_EQ(g.get_weighted_neighbors(2), expected2);
  EXPECT_EQ(g.get_weighted_neighbors(3), expected3);
  EXPECT_EQ(g.get_weighted_neighbors(4), expected4);
}

TEST(GraphTester, GetNodes) {
  cudaqx::graph g;
  g.add_edge(1, 2);
  g.add_edge(2, 3);
  g.add_node(4);
  std::vector<int> expected_nodes = {1, 2, 3, 4};
  std::vector<int> actual_nodes = g.get_nodes();
  std::sort(actual_nodes.begin(), actual_nodes.end());
  EXPECT_EQ(actual_nodes, expected_nodes);
}

TEST(GraphTester, GetEdgeWeight) {
  cudaqx::graph g;
  g.add_edge(1, 2, 1.5);
  g.add_edge(2, 3, 2.5);

  EXPECT_DOUBLE_EQ(g.get_edge_weight(1, 2), 1.5);
  EXPECT_DOUBLE_EQ(g.get_edge_weight(2, 1), 1.5); // Test symmetry
  EXPECT_DOUBLE_EQ(g.get_edge_weight(2, 3), 2.5);
  EXPECT_DOUBLE_EQ(g.get_edge_weight(1, 3), -1.0); // Non-existent edge
}

TEST(GraphTester, UpdateEdgeWeight) {
  cudaqx::graph g;
  g.add_edge(1, 2, 1.5);

  EXPECT_TRUE(g.update_edge_weight(1, 2, 3.0));
  EXPECT_DOUBLE_EQ(g.get_edge_weight(1, 2), 3.0);
  EXPECT_DOUBLE_EQ(g.get_edge_weight(2, 1), 3.0); // Test symmetry

  EXPECT_FALSE(g.update_edge_weight(1, 3, 2.0)); // Non-existent edge
}

TEST(GraphTest, RemoveEdge) {
  cudaqx::graph g;
  g.add_edge(1, 2, 1.0);
  g.add_edge(1, 3, 2.0);
  g.add_edge(2, 3, 3.0);

  g.remove_edge(1, 2);

  EXPECT_EQ(g.get_neighbors(1), std::vector<int>{3});
  EXPECT_EQ(g.get_neighbors(2), std::vector<int>{3});
  std::vector<int> tmp{1, 2};
  EXPECT_EQ(g.get_neighbors(3), tmp);
  EXPECT_EQ(g.num_edges(), 2);
}

TEST(GraphTest, RemoveNode) {
  cudaqx::graph g;
  g.add_edge(1, 2, 1.0);
  g.add_edge(1, 3, 2.0);
  g.add_edge(2, 3, 3.0);
  g.add_edge(3, 4, 1.5);

  g.remove_node(3);

  EXPECT_EQ(g.get_neighbors(1), std::vector<int>{2});
  EXPECT_EQ(g.get_neighbors(2), std::vector<int>{1});
  EXPECT_EQ(g.get_neighbors(4), std::vector<int>{});
  EXPECT_EQ(g.num_nodes(), 3);
  EXPECT_EQ(g.num_edges(), 1);
}

TEST(GraphTest, NumNodes) {
  cudaqx::graph g;
  g.add_edge(1, 2);
  g.add_edge(2, 3);
  g.add_node(4);
  EXPECT_EQ(g.num_nodes(), 4);
}

TEST(GraphTest, NumEdges) {
  cudaqx::graph g;
  g.add_edge(1, 2);
  g.add_edge(2, 3);
  g.add_edge(1, 3);
  EXPECT_EQ(g.num_edges(), 3);
}

TEST(GraphTest, IsConnected) {
  cudaqx::graph g;
  EXPECT_TRUE(g.is_connected()); // Empty graph is considered connected

  g.add_node(1);
  EXPECT_TRUE(g.is_connected()); // Single node graph is connected

  g.add_edge(1, 2);
  g.add_edge(2, 3);
  EXPECT_TRUE(g.is_connected());

  g.add_node(4);
  EXPECT_FALSE(g.is_connected());

  g.add_edge(3, 4);
  EXPECT_TRUE(g.is_connected());
}

TEST(GraphTest, GetDegree) {
  cudaqx::graph g;
  g.add_edge(1, 2);
  g.add_edge(1, 3);
  g.add_edge(1, 4);
  g.add_edge(2, 3);

  EXPECT_EQ(g.get_degree(1), 3);
  EXPECT_EQ(g.get_degree(2), 2);
  EXPECT_EQ(g.get_degree(3), 2);
  EXPECT_EQ(g.get_degree(4), 1);
  EXPECT_EQ(g.get_degree(5), 0); // Non-existent node
}

TEST(GraphTest, MultipleWeightedEdges) {
  cudaqx::graph g;
  g.add_edge(1, 2, 0.5);
  g.add_edge(2, 3, 1.5);
  g.add_edge(3, 1, 2.0);

  EXPECT_EQ(g.num_edges(), 3);
  EXPECT_DOUBLE_EQ(g.get_edge_weight(1, 2), 0.5);
  EXPECT_DOUBLE_EQ(g.get_edge_weight(2, 3), 1.5);
  EXPECT_DOUBLE_EQ(g.get_edge_weight(3, 1), 2.0);

  // Verify neighbors without weights
  std::vector<int> tmp{2, 3}, tmp2{1, 3}, tmp3{1, 2};

  EXPECT_EQ(g.get_neighbors(1), tmp);
  EXPECT_EQ(g.get_neighbors(2), tmp2);
  EXPECT_EQ(g.get_neighbors(3), tmp3);
}

TEST(GraphTest, NegativeWeights) {
  cudaqx::graph g;
  g.add_edge(1, 2, -1.5);
  g.add_edge(2, 3, -0.5);

  EXPECT_DOUBLE_EQ(g.get_edge_weight(1, 2), -1.5);
  EXPECT_DOUBLE_EQ(g.get_edge_weight(2, 3), -0.5);
  std::vector<int> tmp2{1, 3};

  // Verify neighbors without weights
  EXPECT_EQ(g.get_neighbors(1), std::vector<int>{2});
  EXPECT_EQ(g.get_neighbors(2), tmp2);
}

TEST(GraphTest, NodeWeights) {
  cudaqx::graph g;

  // Add nodes with weights
  g.add_node(1, 2.5);
  g.add_node(2, 1.5);
  g.add_edge(1, 2, 1.0);

  // Test node weights
  EXPECT_DOUBLE_EQ(g.get_node_weight(1), 2.5);
  EXPECT_DOUBLE_EQ(g.get_node_weight(2), 1.5);

  // Test default weight
  g.add_node(3);
  EXPECT_DOUBLE_EQ(g.get_node_weight(3), 1.0);

  // Test non-existent node
  EXPECT_DOUBLE_EQ(g.get_node_weight(4), 0.0);

  // Test weight update
  g.set_node_weight(1, 3.0);
  EXPECT_DOUBLE_EQ(g.get_node_weight(1), 3.0);

  // Test node removal
  g.remove_node(1);
  EXPECT_DOUBLE_EQ(g.get_node_weight(1), 0.0);
}

TEST(GraphTest, NodeWeightsClear) {
  cudaqx::graph g;

  g.add_node(1, 2.5);
  g.add_node(2, 1.5);
  g.clear();

  EXPECT_DOUBLE_EQ(g.get_node_weight(1), 0.0);
  EXPECT_DOUBLE_EQ(g.get_node_weight(2), 0.0);
}

TEST(GraphTest, NodeWeightsMultiple) {
  cudaqx::graph g;

  // Add multiple nodes with different weights
  std::vector<std::pair<int, double>> nodes = {
      {1, 1.5}, {2, 2.5}, {3, 3.5}, {4, 4.5}};

  for (const auto &node : nodes) {
    g.add_node(node.first, node.second);
  }

  // Verify all weights
  for (const auto &node : nodes) {
    EXPECT_DOUBLE_EQ(g.get_node_weight(node.first), node.second);
  }
}

TEST(GraphTest, GetDisconnectedVertices) {
  cudaqx::graph g;

  // Add two disconnected components
  g.add_edge(1, 2);
  g.add_edge(2, 3);
  g.add_edge(4, 5);
  g.add_edge(5, 6);

  auto disconnected = g.get_disconnected_vertices();

  std::vector<std::pair<int, int>> expected = {{1, 3}, {1, 4}, {1, 5}, {1, 6},
                                               {2, 4}, {2, 5}, {2, 6}, {3, 4},
                                               {3, 5}, {3, 6}, {4, 6}};

  // Sort both vectors to ensure consistent ordering
  auto sort_pairs = [](std::vector<std::pair<int, int>> &pairs) {
    // First ensure each pair has smaller number first
    for (auto &p : pairs) {
      if (p.first > p.second) {
        std::swap(p.first, p.second);
      }
    }
    // Then sort the vector of pairs
    std::sort(pairs.begin(), pairs.end());
  };

  sort_pairs(disconnected);
  sort_pairs(expected);

  EXPECT_EQ(disconnected, expected);

  // Test with connected graph
  cudaqx::graph g2;
  g2.add_edge(1, 2);
  g2.add_edge(2, 3);
  g2.add_edge(3, 1);

  auto disconnected2 = g2.get_disconnected_vertices();
  EXPECT_TRUE(disconnected2.empty());
}

TEST(GraphTest, EdgeExists) {
  cudaqx::graph g;
  g.add_edge(1, 2);
  EXPECT_TRUE(g.edge_exists(1, 2));
  EXPECT_TRUE(g.edge_exists(2, 1));
  EXPECT_FALSE(g.edge_exists(1, 3));
  EXPECT_FALSE(g.edge_exists(3, 4));
}
