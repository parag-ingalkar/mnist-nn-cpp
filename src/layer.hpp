// layer.hpp
#pragma once

#include "tensor.hpp"
#include "matvec.hpp"
#include <cmath>
#include <algorithm>
#include <random>

class Layer
{
public:
    virtual ~Layer() = default;
    virtual Tensor<double> forward(const Tensor<double> &input) = 0;
    virtual Tensor<double> backward(const Tensor<double> &error, double learning_rate) = 0;
};

class FullyConnectedLayer : public Layer
{
public:
    FullyConnectedLayer(size_t input_size, size_t output_size)
        : weights_(output_size, input_size), bias_(output_size)
    {

        // Kaiming initialization for better convergence
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> d(0.0, std::sqrt(2.0 / input_size));

        for (size_t i = 0; i < output_size; ++i)
        {
            for (size_t j = 0; j < input_size; ++j)
            {
                weights_(i, j) = d(gen);
            }
            bias_(i) = 0.0; // Initialize bias to zero
        }
    }

    Tensor<double> forward(const Tensor<double> &input) override
    {
        input_ = input;

        // Create vector from input tensor
        Vector<double> input_vec(input.numElements());
        for (size_t i = 0; i < input.numElements(); ++i)
        {
            input_vec(i) = input({i});
        }

        // Matrix-vector multiplication (y = Wx + b)
        Vector<double> output_vec = matvec(weights_, input_vec);

        // Add bias
        for (size_t i = 0; i < output_vec.size(); ++i)
        {
            output_vec(i) += bias_(i);
        }

        // Convert result back to tensor
        Tensor<double> output({output_vec.size()});
        for (size_t i = 0; i < output_vec.size(); ++i)
        {
            output({i}) = output_vec(i);
        }

        return output;
    }

    Tensor<double> backward(const Tensor<double> &error, double learning_rate) override
    {
        // Convert error tensor to vector
        Vector<double> error_vec(error.numElements());
        for (size_t i = 0; i < error.numElements(); ++i)
        {
            error_vec(i) = error({i});
        }

        // Convert input tensor to vector
        Vector<double> input_vec(input_.numElements());
        for (size_t i = 0; i < input_.numElements(); ++i)
        {
            input_vec(i) = input_({i});
        }

        // Update weights gradients (W = W - η * E * X^T)
        for (size_t i = 0; i < weights_.rows(); ++i)
        {
            for (size_t j = 0; j < weights_.cols(); ++j)
            {
                weights_(i, j) -= learning_rate * error_vec(i) * input_vec(j);
            }
            // Update bias (b = b - η * E)
            bias_(i) -= learning_rate * error_vec(i);
        }

        // Compute error for previous layer (E_prev = W^T * E)
        Matrix<double> weights_t(weights_.cols(), weights_.rows());
        for (size_t i = 0; i < weights_.rows(); ++i)
        {
            for (size_t j = 0; j < weights_.cols(); ++j)
            {
                weights_t(j, i) = weights_(i, j);
            }
        }

        Vector<double> prev_error = matvec(weights_t, error_vec);

        // Convert to tensor
        Tensor<double> prev_error_tensor({prev_error.size()});
        for (size_t i = 0; i < prev_error.size(); ++i)
        {
            prev_error_tensor({i}) = prev_error(i);
        }

        return prev_error_tensor;
    }

private:
    Matrix<double> weights_;
    Vector<double> bias_;
    Tensor<double> input_;
};

class ReLULayer : public Layer
{
public:
    Tensor<double> forward(const Tensor<double> &input) override
    {
        input_ = input;
        Tensor<double> output(input.shape());

        for (size_t i = 0; i < input.numElements(); ++i)
        {
            output({i}) = std::max(0.0, input({i}));
        }

        return output;
    }

    Tensor<double> backward(const Tensor<double> &error, double learning_rate) override
    {
        Tensor<double> output(error.shape());

        for (size_t i = 0; i < error.numElements(); ++i)
        {
            // ReLU derivative: 1 if input > 0, 0 otherwise
            output({i}) = (input_({i}) > 0) ? error({i}) : 0.0;
        }

        return output;
    }

private:
    Tensor<double> input_;
};

class SoftmaxLayer : public Layer
{
public:
    Tensor<double> forward(const Tensor<double> &input) override
    {
        Tensor<double> output(input.shape());

        // Find maximum value for numerical stability
        double max_val = input({0});
        for (size_t i = 1; i < input.numElements(); ++i)
        {
            max_val = std::max(max_val, input({i}));
        }

        // Compute exp(x - max) for each element
        double sum = 0.0;
        for (size_t i = 0; i < input.numElements(); ++i)
        {
            output({i}) = std::exp(input({i}) - max_val);
            sum += output({i});
        }

        // Normalize
        for (size_t i = 0; i < output.numElements(); ++i)
        {
            output({i}) /= sum;
        }

        output_ = output;
        return output;
    }

    Tensor<double> backward(const Tensor<double> &error, double learning_rate) override
    {
        // For softmax with cross-entropy loss, the derivative is simplified
        // This implementation assumes the error is already in the right format
        return error;
    }

private:
    Tensor<double> output_;
};