#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>

class Config
{
public:
    explicit Config(const std::string &filename)
    {
        loadConfig(filename);
    }

    template <typename T>
    T get_value(const std::string &key) const
    {
        auto it = configData.find(key);
        if (it == configData.end())
        {
            throw std::runtime_error("Key not found: " + key);
        }
        return convert<T>(it->second);
    }

    void print_config() const
    {
        std::cout << "Configuration Settings:\n";
        for (const auto &pair : configData)
        {
            std::cout << pair.first << " = " << pair.second << "\n";
        }
    }

private:
    std::unordered_map<std::string, std::string> configData;

    void loadConfig(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Unable to open config file: " + filename);
        }

        std::string line;
        while (std::getline(file, line))
        {
            if (line.empty() || line[0] == '#')
                continue; // Ignore empty lines and comments
            size_t pos = line.find('=');
            if (pos == std::string::npos)
                continue; // Skip invalid lines

            std::string key = trim(line.substr(0, pos));
            std::string value = trim(line.substr(pos + 1));

            configData[key] = value;
        }
        file.close();
    }

    static std::string trim(const std::string &str)
    {
        size_t first = str.find_first_not_of(" \t");
        if (first == std::string::npos)
            return "";
        size_t last = str.find_last_not_of(" \t");
        return str.substr(first, (last - first + 1));
    }

    template <typename T>
    static T convert(const std::string &value);
};

// Specializations for different types
template <>
int Config::convert<int>(const std::string &value)
{
    return std::stoi(value);
}

template <>
float Config::convert<float>(const std::string &value)
{
    return std::stof(value);
}

template <>
double Config::convert<double>(const std::string &value)
{
    return std::stod(value);
}

template <>
std::string Config::convert<std::string>(const std::string &value)
{
    return value;
}
