// network.hpp
#pragma once

#include "layer.hpp"
#include "loss.hpp"
#include <vector>
#include <memory>

class NeuralNetwork
{
public:
    NeuralNetwork(size_t input_size, size_t hidden_size, size_t num_classes)
    {

        // Add fully connected layer (input -> hidden)
        layers_.push_back(std::make_unique<FullyConnectedLayer>(input_size, hidden_size));

        // Add ReLU activation
        layers_.push_back(std::make_unique<ReLULayer>());

        // Add fully connected layer (hidden -> output)
        layers_.push_back(std::make_unique<FullyConnectedLayer>(hidden_size, num_classes));

        // Add softmax activation
        layers_.push_back(std::make_unique<SoftmaxLayer>());

        // Set loss function
        loss_ = std::make_unique<CrossEntropyLoss>();
    }

    Tensor<double> forward(const Tensor<double> &input) const
    {
        Tensor<double> current = input;
        for (auto &layer : layers_)
        {
            std::cout << "Layer forward... \n";
            current = layer->forward(current);
        }
        return current;
    }

    void backward(const Tensor<double> &prediction, const Tensor<double> &target, double learning_rate)
    {
        // Calculate initial error gradient from loss function
        Tensor<double> error = loss_->gradient(prediction, target);

        // Propagate error backwards through the network
        for (int i = layers_.size() - 1; i >= 0; i--)
        {
            std::cout << "Backward error " << layers_.size() << std::endl;
            error = layers_[i]->backward(error, learning_rate);
        }
    }

    double compute_loss(const Tensor<double> &prediction, const Tensor<double> &target)
    {
        return loss_->compute(prediction, target);
    }

    size_t predict(const Tensor<double> &input) const
    {
        Tensor<double> output = forward(input);

        // Find the index of the highest probability
        size_t max_idx = 0;
        double max_val = output({0});

        for (size_t i = 1; i < output.numElements(); i++)
        {
            if (output({i}) > max_val)
            {
                max_val = output({i});
                max_idx = i;
            }
        }

        return max_idx;
    }

    void train(std::vector<Tensor<double>> inputs, std::vector<Tensor<double>> targets, int epochs, double learning_rate)
    {
        std::cout << "Starting Training\n";
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            std::cout << "Starting Training for epoch " << epoch << std::endl;
            double total_loss = 0.0;
            for (size_t i = 0; i < inputs.size(); i++)
            {
                std::cout << "i = " << i << "\nInput SIze = " << inputs.size() << std::endl;
                Tensor<double> prediction = forward(inputs[i]);
                total_loss += compute_loss(prediction, targets[i]);
                backward(prediction, targets[i], learning_rate);
            }
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / inputs.size() << std::endl;
        }
    }

    double test(const std::vector<Tensor<double>> &test_inputs,
                const std::vector<Tensor<double>> &test_targets)
    {
        double total_loss = 0.0;
        int correct_predictions = 0;

        for (size_t i = 0; i < test_inputs.size(); ++i)
        {
            Tensor<double> prediction = forward(test_inputs[i]);

            // Compute loss (but do not call backward)
            total_loss += compute_loss(prediction, test_targets[i]);

            // Convert softmax output to predicted class (argmax)
            const std::vector<double> &pred_data = prediction.getData();
            size_t predicted_class = std::distance(pred_data.begin(), std::max_element(pred_data.begin(), pred_data.end()));

            const std::vector<double> &target_data = test_targets[i].getData();
            size_t actual_class = std::distance(target_data.begin(),
                                                std::max_element(target_data.begin(), target_data.end()));

            if (predicted_class == actual_class)
            {
                correct_predictions++;
            }
        }

        double avg_loss = total_loss / test_inputs.size();
        double accuracy = (double)correct_predictions / test_inputs.size();

        std::cout << "Test Loss: " << avg_loss << std::endl;
        std::cout << "Test Accuracy: " << accuracy * 100.0 << "%" << std::endl;

        return accuracy; // Returning accuracy for further use if needed
    }

private:
    std::vector<std::unique_ptr<Layer>> layers_;
    std::unique_ptr<Loss> loss_;
};