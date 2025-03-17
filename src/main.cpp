#include <iostream>
#include <filesystem>
#include "config.hpp"
#include "data_loader.hpp"
#include "tensor.hpp"
#include "network.hpp"

namespace fs = std::filesystem;

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

int main(int argc, char *argv[])
{

    Config config("mnist-configs/input.config");
    config.print_config();

    // Get file paths from config
    std::string rel_path_test_labels = config.get_value<std::string>("rel_path_test_labels");
    std::string rel_path_test_images = config.get_value<std::string>("rel_path_test_images");
    std::string rel_path_train_images = config.get_value<std::string>("rel_path_train_images");
    std::string rel_path_train_labels = config.get_value<std::string>("rel_path_train_labels");
    std::string rel_path_log_file = config.get_value<std::string>("rel_path_log_file");
    int num_epochs = config.get_value<int>("num_epochs");
    int batch_size = config.get_value<int>("batch_size");
    int hidden_size = config.get_value<int>("hidden_size");
    double learning_rate = config.get_value<double>("learning_rate");

    // Load training data
    std::cout << "Loading training images from " << rel_path_train_images << std::endl;
    std::vector<Tensor<double>> train_images = readImagesFromFile(rel_path_train_images);
    std::cout << "Loading training labels from " << rel_path_train_labels << std::endl;
    std::vector<Tensor<double>> train_labels = readLabelsFromFile(rel_path_train_labels);

    // Load test data
    std::cout << "Loading test images from " << rel_path_test_images << std::endl;
    std::vector<Tensor<double>> test_images = readImagesFromFile(rel_path_test_images);
    std::cout << "Loading test labels from " << rel_path_test_labels << std::endl;
    std::vector<Tensor<double>> test_labels = readLabelsFromFile(rel_path_test_labels);

    std::cout << train_images[0] << std::endl;
    std::cout << train_labels[0] << std::endl;

    // // Flatten image tensors
    // std::vector<Tensor<double>> flattened_train_images;
    // for (const auto &image : train_images)
    // {
    //     flattened_train_images.push_back(flattenImage(image));
    // }

    // std::vector<Tensor<double>> flattened_test_images;
    // for (const auto &image : test_images)
    // {
    //     flattened_test_images.push_back(flattenImage(image));
    // }

    // // Create neural network
    // size_t input_size = flattened_train_images[0].numElements(); // 28*28 = 784
    // size_t output_size = 10;                                     // 10 digits (0-9)

    // std::cout << "Creating neural network with architecture: "
    //           << input_size << " -> " << hidden_size << " -> " << output_size << std::endl;
    // NeuralNetwork network(input_size, hidden_size, output_size);

    // // Train the network
    // network.train(flattened_train_images, train_labels, num_epochs, learning_rate);

    // // Test the network
    // double accuracy = network.test(flattened_test_images, test_labels);

    // std::cout << "Test accuracy: " << accuracy * 100.0 << "%" << std::endl;

    return 0;
}