#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>

#include "config.hpp"
#include "tensor.hpp"
#include "network.hpp"
#include "data_loader.hpp"

// Function to flatten a 2D image tensor into a 1D tensor
Tensor<double> flattenImage(const Tensor<double> &image)
{
    std::vector<size_t> shape = image.shape();
    size_t rows = shape[0];
    size_t cols = shape[1];
    size_t total_size = rows * cols;

    Tensor<double> flattened({total_size});

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            flattened({i * cols + j}) = image({i, j});
        }
    }

    return flattened;
}

// Function to convert a scalar label to a one-hot encoded tensor
Tensor<double> oneHotEncode(uint8_t label, size_t num_classes = 10)
{
    Tensor<double> one_hot({num_classes}, 0.0);
    one_hot({label}) = 1.0;
    return one_hot;
}

// Function to load multiple images and labels
std::pair<std::vector<Tensor<double>>, std::vector<Tensor<double>>> loadDataset(
    const std::string &images_file,
    const std::string &labels_file,
    size_t num_samples = 0)
{

    std::ifstream images(images_file, std::ios::binary);
    std::ifstream labels(labels_file, std::ios::binary);

    if (!images.is_open() || !labels.is_open())
    {
        std::cerr << "Error: Cannot open dataset files" << std::endl;
        exit(1);
    }

    // Read image file header
    uint32_t magic_number, num_images, num_rows, num_cols;
    images.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    images.read(reinterpret_cast<char *>(&num_images), sizeof(num_images));
    images.read(reinterpret_cast<char *>(&num_rows), sizeof(num_rows));
    images.read(reinterpret_cast<char *>(&num_cols), sizeof(num_cols));

    magic_number = toSystemEndian(magic_number);
    num_images = toSystemEndian(num_images);
    num_rows = toSystemEndian(num_rows);
    num_cols = toSystemEndian(num_cols);

    // Read label file header
    uint32_t label_magic, num_labels;
    labels.read(reinterpret_cast<char *>(&label_magic), sizeof(label_magic));
    labels.read(reinterpret_cast<char *>(&num_labels), sizeof(num_labels));

    label_magic = toSystemEndian(label_magic);
    num_labels = toSystemEndian(num_labels);

    // Limit number of samples if specified
    if (num_samples > 0 && num_samples < num_images)
    {
        num_images = num_samples;
    }

    std::cout << "Loading " << num_images << " images..." << std::endl;

    std::vector<Tensor<double>> image_tensors;
    std::vector<Tensor<double>> label_tensors;

    size_t image_size = num_rows * num_cols;
    std::vector<uint8_t> image_data(image_size);

    for (size_t i = 0; i < num_images; ++i)
    {
        // Read image data
        images.read(reinterpret_cast<char *>(image_data.data()), image_size);

        // Create image tensor
        Tensor<double> image({num_rows, num_cols});
        for (size_t pixel = 0; pixel < image_size; ++pixel)
        {
            size_t row = pixel / num_cols;
            size_t col = pixel % num_cols;
            image({row, col}) = static_cast<double>(image_data[pixel]) / 255.0;
        }

        // Flatten image
        Tensor<double> flattened = flattenImage(image);
        image_tensors.push_back(flattened);

        // Read label
        uint8_t label;
        labels.read(reinterpret_cast<char *>(&label), 1);

        // One-hot encode label
        Tensor<double> one_hot = oneHotEncode(label);
        label_tensors.push_back(one_hot);

        // Print progress
        if ((i + 1) % 1000 == 0 || i == num_images - 1)
        {
            std::cout << "Loaded " << (i + 1) << "/" << num_images << " images" << std::endl;
        }
    }

    images.close();
    labels.close();

    return {image_tensors, label_tensors};
}

// Function to compute accuracy
double computeAccuracy(const NeuralNetwork &network,
                       const std::vector<Tensor<double>> &images,
                       const std::vector<Tensor<double>> &labels)
{
    size_t correct = 0;

    for (size_t i = 0; i < images.size(); ++i)
    {
        size_t predicted = network.predict(images[i]);

        // Find the index of the highest value in the one-hot encoded label
        size_t actual = 0;
        double max_val = labels[i]({0});

        for (size_t j = 1; j < 10; ++j)
        {
            if (labels[i]({j}) > max_val)
            {
                max_val = labels[i]({j});
                actual = j;
            }
        }

        if (predicted == actual)
        {
            correct++;
        }
    }

    return static_cast<double>(correct) / images.size();
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    // Load configuration
    Config config(argv[1]);

    // Get configuration parameters
    std::string train_images_file = config.get_string("train_images_file");
    std::string train_labels_file = config.get_string("train_labels_file");
    std::string test_images_file = config.get_string("test_images_file");
    std::string test_labels_file = config.get_string("test_labels_file");

    size_t input_size = config.get_int("input_size", 784); // 28x28 pixels
    size_t hidden_size = config.get_int("hidden_size", 128);
    size_t output_size = config.get_int("output_size", 10); // 10 digits

    double learning_rate = config.get_double("learning_rate", 0.01);
    size_t num_epochs = config.get_int("num_epochs", 10);
    size_t batch_size = config.get_int("batch_size", 64);

    size_t train_samples = config.get_int("train_samples", 60000);
    size_t test_samples = config.get_int("test_samples", 10000);

    // Load training data
    std::cout << "Loading training data..." << std::endl;
    auto [train_images, train_labels] = loadDataset(train_images_file, train_labels_file, train_samples);

    // Load test data
    std::cout << "Loading test data..." << std::endl;
    auto [test_images, test_labels] = loadDataset(test_images_file, test_labels_file, test_samples);

    // Initialize neural network
    std::cout << "Initializing neural network..." << std::endl;
    NeuralNetwork network(input_size, hidden_size, output_size);

    // Training loop
    std::cout << "Starting training..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t epoch = 0; epoch < num_epochs; ++epoch)
    {
        double total_loss = 0.0;

        // Create indices for shuffling
        std::vector<size_t> indices(train_images.size());
        for (size_t i = 0; i < indices.size(); ++i)
        {
            indices[i] = i;
        }

        // Shuffle indices
        std::random_shuffle(indices.begin(), indices.end());

        // Process in batches
        for (size_t batch_start = 0; batch_start < train_images.size(); batch_start += batch_size)
        {
            size_t batch_end = std::min(batch_start + batch_size, train_images.size());

            for (size_t i = batch_start; i < batch_end; ++i)
            {
                size_t idx = indices[i];

                // Forward pass
                Tensor<double> prediction = network.forward(train_images[idx]);

                // Compute loss
                double loss = network.compute_loss(prediction, train_labels[idx]);
                total_loss += loss;

                // Backward pass and update weights
                network.backward(prediction, train_labels[idx], learning_rate);
            }
        }

        // Compute average loss
        double avg_loss = total_loss / train_images.size();

        // Compute accuracy on training data
        double train_accuracy = computeAccuracy(network, train_images, train_labels);

        // Print progress
        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs
                  << ", Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << ", Train Accuracy: " << std::fixed << std::setprecision(4) << train_accuracy << std::endl;

        // Evaluate on test data every few epochs
        if ((epoch + 1) % 5 == 0 || epoch == num_epochs - 1)
        {
            double test_accuracy = computeAccuracy(network, test_images, test_labels);
            std::cout << "Test Accuracy: " << std::fixed << std::setprecision(4) << test_accuracy << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "Training completed in " << duration.count() << " seconds" << std::endl;

    // Final evaluation on test data
    double final_test_accuracy = computeAccuracy(network, test_images, test_labels);
    std::cout << "Final Test Accuracy: " << std::fixed << std::setprecision(4) << final_test_accuracy << std::endl;

    return 0;
}