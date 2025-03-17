#include <iostream>
#include <fstream>
#include <vector>
#include <bit>     // For std::byteswap (C++23)
#include <cstdint> // For uint32_t, uint8_t
#include "tensor.hpp"

// Function to check if the system is little-endian
bool isLittleEndian()
{
    return (std::endian::native == std::endian::little ? 1 : 0);
}

uint32_t toSystemEndian(uint32_t value)
{
    return (isLittleEndian() ? std::byteswap(value) : value);
}

std::vector<Tensor<double>> readImagesFromFile(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        std::exit(1); // Exit program if file opening fails
    }

    uint32_t magic_number, num_images, num_rows, num_cols;

    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char *>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char *>(&num_rows), sizeof(num_rows));
    file.read(reinterpret_cast<char *>(&num_cols), sizeof(num_cols));

    magic_number = toSystemEndian(magic_number);
    num_images = toSystemEndian(num_images);
    num_rows = toSystemEndian(num_rows);
    num_cols = toSystemEndian(num_cols);

    // Validate the magic number (should be 2051 for MNIST images)
    if (magic_number != 2051)
    {
        std::cerr << "Error: Invalid magic number in file " << filename << std::endl;
        std::exit(1);
    }

    // Verify reasonable values for dimensions
    if (num_rows == 0 || num_cols == 0 || num_images == 0 || num_images > 100000)
    {
        std::cerr << "Error: Invalid dimensions in file " << filename << std::endl;
        std::exit(1);
    }

    // Print dataset information
    std::cout << "Magic Number: " << magic_number << std::endl;
    std::cout << "Number of Images: " << num_images << std::endl;
    std::cout << "Image Size: " << num_rows << "x" << num_cols << std::endl;

    size_t image_size = num_rows * num_cols;

    std::vector<Tensor<double>> image_tensors(num_images);
    std::vector<u_int8_t> image_data(image_size);

    for (size_t i = 0; i < num_images; i++)
    {
        file.read(reinterpret_cast<char *>(image_data.data()), image_size);

        // Check if read was successful
        if (!file)
        {
            std::cerr << "Error: Failed to read image data at index " << i << std::endl;
            std::exit(1);
        }

        Tensor<double> tensor({num_rows, num_cols});
        for (auto pixel = 0; pixel < image_size; pixel++)
        {
            size_t row = pixel / num_cols;
            size_t col = pixel % num_cols;
            // Convert uint8_t to double and normalize to [0.0, 1.0] range in one step
            tensor({row, col}) = static_cast<double>(image_data[pixel]) / 255.0;
        }
        image_tensors[i] = tensor;
    }

    file.close();
    return image_tensors;
}

std::vector<Tensor<double>> readLabelsFromFile(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        std::exit(1); // Exit program if file opening fails
    }

    uint32_t magic_number, num_items;

    file.read(reinterpret_cast<char *>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char *>(&num_items), sizeof(num_items));

    magic_number = toSystemEndian(magic_number);
    num_items = toSystemEndian(num_items);

    // Validate the magic number (should be 2049 for MNIST labels)
    if (magic_number != 2049)
    {
        std::cerr << "Error: Invalid magic number in file " << filename << std::endl;
        std::exit(1);
    }

    // Verify reasonable values for number of items
    if (num_items == 0 || num_items > 100000)
    {
        std::cerr << "Error: Invalid number of items in file " << filename << std::endl;
        std::exit(1);
    }

    // Print dataset information
    std::cout << "Magic Number: " << magic_number << std::endl;
    std::cout << "Number of Labels: " << num_items << std::endl;

    size_t num_classes = 10;

    std::vector<Tensor<double>> label_tensors(num_items);
    std::vector<u_int8_t> labels_data(1); // Read one label at a time

    for (size_t i = 0; i < num_items; i++)
    {
        file.read(reinterpret_cast<char *>(labels_data.data()), 1);

        // Check if read was successful
        if (!file)
        {
            std::cerr << "Error: Failed to read label data at index " << i << std::endl;
            std::exit(1);
        }

        // Validate the label value
        if (labels_data[0] >= num_classes)
        {
            std::cerr << "Error: Invalid label value: " << static_cast<int>(labels_data[0]) << std::endl;
            std::exit(1);
        }

        Tensor<double> tensor({num_classes}, 0.0); // Initialize with all zeros
        tensor({labels_data[0]}) = 1.0;            // Set the position of the label to 1.0

        label_tensors[i] = tensor;
    }

    file.close();
    return label_tensors;
}