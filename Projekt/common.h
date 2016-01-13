#ifndef common_h
#define common_h

#include <iostream>
#include <fstream>
#include <climits>
#include <cmath>
#include <vector>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <set>

namespace common
{
    class BinarySerializable
    {
    public:
        unsigned int SerilizedSize = 0;
        
        unsigned char* serialize()
        {
            auto buffer = new unsigned char[SerilizedSize];
            serialize(buffer);
            return buffer;
        }
        virtual void serialize(unsigned char* buffer) = 0;
    };
}

#endif 