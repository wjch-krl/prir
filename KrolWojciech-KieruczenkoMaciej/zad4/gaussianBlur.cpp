#include <iostream>

#include <mpi.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define MASK_SIZE 7

int weights[MASK_SIZE][MASK_SIZE] = {
    {1,1,2,2,2,1,1},
    {1,2,2,4,2,2,1},
    {2,2,4,8,4,2,2},
    {2,4,8,16,8,4,2},
    {2,2,4,8,4,2,2},
    {1,2,2,4,2,2,1},
    {1,1,2,2,2,1,1}
};;

    uchar* Blur(uchar* image, int width, int height, int workHeight)
    {
        uchar* resultImage = new uchar[width+width*height];
        int offset = MASK_SIZE / 2;
        for(int i= 0; i< width; i++)
        {
            for (int j = 0; j< workHeight; j++)
            {
                int pixelValue = 0;
                //[i][j] current pixel
                int weightsIdx = 0;
                int weightsIdy = 0;
                int weightsSum = 0;
                for(weightsIdx = (i - offset) > 0 ? 0 : offset- i;
                    weightsIdx<MASK_SIZE; weightsIdx ++)
                {
                    for(weightsIdy = (j - offset) > 0 ? 0 : offset- j;
                        weightsIdy<MASK_SIZE; weightsIdy++)
                    {
                        int k = weightsIdx - offset + i;
                        int l = weightsIdy - offset + j;
                        if(k<width && l<height)
                        {
                            pixelValue += weights[weightsIdx][weightsIdy]*image[k+width*l];
                            weightsSum += weights[weightsIdx][weightsIdy];
                        }
                    }
                }
                pixelValue /= weightsSum;
                if (pixelValue > 255)
                    pixelValue = 255;
                else if (pixelValue < 0)
                    pixelValue = 0;
                resultImage[i+width*j] = (uchar)pixelValue;
            }
        }
        return resultImage;
    }

uchar** matToArrays(cv::Mat& mat)
{
    int chanels = mat.channels();
    int cols = mat.cols;
    int rows = mat.rows;
    uchar **m = new uchar*[chanels];
    int chanelSize = cols + cols*rows;
    for(int k = 0 ; k< chanels ; k++)
    {
        m[k] = new uchar[chanelSize];
    }
    for (int i = 0; i < cols; ++i)
    {
        for (int j = 0; j < rows; ++j)
        {
            cv::Vec3b point = mat.at<cv::Vec3b>(cv::Point(i,j));
            for(int k = 0 ; k< chanels ; k++)
            {
                uchar color = point[k];
                m[k][i + cols * j] = color;
            }
        }
    }
    return m;
}

cv::Mat* arraysToMat(uchar** m,int chanelCount, int height, int widht)
{
    cv:: Mat* mat = new cv::Mat(height,widht,CV_8UC(chanelCount));
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < widht; ++j)
        {
            cv::Vec3b point;
            for(int k = 0 ; k< chanelCount ; k++)
            {
                point[k] = m[k][j+widht * i];
            }
            mat->at<cv::Vec3b>(cv::Point(j,i)) = point;
        }
    }
    return mat;
}


int main(int argc, char** argv)
{
    //GaussianBlur* gb = new GaussianBlur();
    // Initialize MPI
    MPI_Init(NULL, NULL);

    // Get number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get process rank
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int chanel;
    if (world_rank == 0) 
    {
        cv::Mat inputImage = cv::imread("/Users/wojciechkrol/tmp/dupxo/test.jpg", 1 );
        int chanelCount =inputImage.channels();
        uchar** imageArrays = matToArrays(inputImage);
        uchar** bluredArrays = new uchar*[chanelCount];

        for (int i = 0; i <chanelCount;i++)
        {
            MPI_Send(&i, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);            
            bluredArrays[i] = Blur(imageArrays[i], inputImage.cols, inputImage.rows, inputImage.rows);
        }
        
        cv::Mat* bluredImage = arraysToMat(bluredArrays, chanelCount, inputImage.rows, inputImage.cols);      
        imwrite("/Users/wojciechkrol/tmp/dupxo/blured.jpg", *bluredImage);
    } 
    else
    {
        MPI_Recv(&chanel, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
    }
    
    //Cleanup
    MPI_Finalize();
}
