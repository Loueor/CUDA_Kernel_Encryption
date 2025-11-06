#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

int main() {
    const std::string filename = "./testapp/testapp.cubin";
    const std::string output_filename = "./cubin_bytes.txt";
    
    // Open file in binary mode
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return 1;
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Read entire file into buffer
    std::vector<unsigned char> buffer(fileSize);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();
    
    // Open output file for writing
    std::ofstream output_file(output_filename);
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not create output file " << output_filename << std::endl;
        return 1;
    }
    
    // Output to both console and file
    std::cout << "File: " << filename << " (" << fileSize << " bytes)" << std::endl;
    output_file << "File: " << filename << " (" << fileSize << " bytes)" << std::endl;
    
    std::cout << "Hex dump:" << std::endl;
    output_file << "Hex dump:" << std::endl;
    
    // Save original stream flags
    std::ios_base::fmtflags console_flags = std::cout.flags();
    
    // Set hex format for both streams
    std::cout << std::hex << std::setfill('0');
    output_file << std::hex << std::setfill('0');
    
    // Print hex dump to both console and file
    for (size_t i = 0; i < fileSize; ++i) {
        // Print byte in hex format
        std::cout << std::setw(2) << static_cast<unsigned int>(buffer[i]) << " ";
        output_file << std::setw(2) << static_cast<unsigned int>(buffer[i]) << " ";
        
        // New line every 16 bytes
        if ((i + 1) % 16 == 0) {
            std::cout << std::endl;
            output_file << std::endl;
        }
    }
    
    // Add final newline if file doesn't end on a 16-byte boundary
    if (fileSize % 16 != 0) {
        std::cout << std::endl;
        output_file << std::endl;
    }
    
    // Restore console flags
    std::cout.flags(console_flags);
    
    output_file.close();
    std::cout << "\nOutput also saved to: " << output_filename << std::endl;
    
    return 0;
}