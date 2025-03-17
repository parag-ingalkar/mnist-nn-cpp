// loss.hpp
#pragma once

#include "tensor.hpp"
#include <cmath>

class Loss
{
public:
    virtual ~Loss() = default;
    virtual double compute(const Tensor<double> &prediction, const Tensor<double> &target) = 0;
    virtual Tensor<double> gradient(const Tensor<double> &prediction, const Tensor<double> &target) = 0;
};

class CrossEntropyLoss : public Loss
{
public:
    double compute(const Tensor<double> &prediction, const Tensor<double> &target) override
    {
        double loss = 0.0;

        for (size_t i = 0; i < prediction.numElements(); ++i)
        {
            // Add small epsilon to avoid log(0)
            double epsilon = 1e-10;
            loss -= target({i}) * std::log(prediction({i}) + epsilon);
        }

        return loss;
    }

    Tensor<double> gradient(const Tensor<double> &prediction, const Tensor<double> &target) override
    {
        // For softmax + cross-entropy, the gradient simplifies to (prediction - target)
        Tensor<double> grad(prediction.shape());

        for (size_t i = 0; i < prediction.numElements(); ++i)
        {
            grad({i}) = prediction({i}) - target({i});
        }

        return grad;
    }
};