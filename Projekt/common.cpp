//
// Created by Wojciech Król on 24.01.2016.
//

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