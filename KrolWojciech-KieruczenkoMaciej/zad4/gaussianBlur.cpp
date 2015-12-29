#include <iostream>

int weights[7][7] = {
            {1,1,2,2,2,1,1},
            {1,2,2,4,2,2,1},
            {2,2,4,8,4,2,2},
            {2,4,8,16,8,4,2},
            {2,2,4,8,4,2,2},
            {1,2,2,4,2,2,1},
            {1,1,2,2,2,1,1}
        };;

class GaussianBlur
{
private:

    int size;
    int weightsSum;
    
public:
    
    GaussianBlur()
    {
        this->size = 7;
        this->weightsSum = 0;           
        for(int i = 0;i<size;i++)
        {
            for(int j = 0;j<size;j++)
            {
                this->weightsSum += weights[i][j];
            }
        }
    }

    unsigned char** Blur(unsigned char** image, int width, int height)
    {
        unsigned char** resultImage = new unsigned char*[width];
        int offset = this->size / 2 + 1;
        for(int i= 0; i< width; i++)
        {
            resultImage[i] = new unsigned char[height];
            for (int j = 0; j< height; j++)
            {
                int pixelValue = 0;
                //[i][j] current pixel
                int weightsIdx = 0;
                int weightsIdy = 0;
                
                for(weightsIdx = (i - this->size) > 0 ? 0 : this->size - i; 
                    weightsIdx<this->size && (i + weightsIdx) < width; weightsIdx ++)
                {
                    for(weightsIdx = (i - this->size) > 0 ? 0 : this->size - i;
                        weightsIdx<this->size && (i + weightsIdx) < width; weightsIdx ++)
                    {
                        int k = weightsIdx - offset + i;
                        int l = weightsIdy - offset + j;                    
                        pixelValue += weights[weightsIdx][weightsIdy]*resultImage[k][l];
                    }
                }  
                pixelValue /= weightsSum;
                if (pixelValue > 255)
                    pixelValue = 255;
                else if (pixelValue < 0)
                    pixelValue = 0;           
                resultImage[i][j] = (unsigned char)pixelValue;
            }
        }
        return resultImage;
    }    
};

void printMatrix(unsigned char** mtrx, int size)
{
    for(int i = 0;i<size;i++)
    {
        for(int j = 0;j<size;j++)
        {
            std::cout<<mtrx[i][j] << "\t";
        }
        std::cout <<"\n";
    }
}

int main(int argc, char** argv)
{
    GaussianBlur* gb = new GaussianBlur();
    int size = 7;
    unsigned char test1[7][7] = {
                {1,1,2,2,2,1,1},
                {1,2,2,4,2,2,1},
                {2,2,4,8,4,2,2},
                {2,4,8,16,8,4,2},
                {2,2,4,8,4,2,2},
                {1,2,2,4,2,2,1},
                {1,1,2,2,2,1,1}
            };
    unsigned char test2[7][7] = {
                {1,1,1,1,1,1,1},
                {1,1,1,1,1,1,1},
                {1,1,1,1,1,1,1},
                {1,1,1,1,1,1,1},
                {1,1,1,1,1,1,1},
                {1,1,1,1,1,1,1},
                {1,1,1,1,1,1,1}
            };
            
    unsigned char test3[7][7] = {
                {1,2,1,2,1,2,1},
                {2,1,2,1,2,1,2},
                {1,2,1,2,1,2,1},
                {2,1,2,1,2,1,2},
                {1,2,1,2,1,2,1},
                {2,1,2,1,2,1,2},
                {1,2,1,2,1,2,1}
            };
    unsigned char ** tmp = new  unsigned char* [size];  
     for(int i = 0;i<size;i++)
    {
        tmp[i] = new unsigned char[size];
        for(int j = 0;j<size;j++)
        {
            tmp[i][j]=test1[i][j];
        }
    }
    printMatrix(tmp,size);
    std::cout<<"\n======\n";
    printMatrix(gb->Blur(tmp,size,size),size);       
}
