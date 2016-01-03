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
            {
                pixelValue = 255;
            }
            else if (pixelValue < 0)
            {
                pixelValue = 0;
            }
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
    if(argc != 3)
    {
        std::cout<<"Usage: \nmpirun "<<argv[0]<< " INPUT_FILE_PATH OUTPUT_FILE_PATH \n";
        exit(EXIT_FAILURE);
    }
    //GaussianBlur* gb = new GaussianBlur();
    // Initialize MPI
    MPI_Init(NULL, NULL);

    // Get number of processes
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Get process rank
    int worldRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    if (worldRank == 0) 
    {
        cv::Mat inputImage = cv::imread(argv[1], 1 );
        if(! inputImage.data )                             
        {
            std::cout << "Invalid image \n";
            MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);            
            exit(EXIT_FAILURE);
        }
        int chanelCount =inputImage.channels();
        int imgWidth = inputImage.cols;
        int imgHeight = inputImage.rows;
        int fragmentHeight = inputImage.rows / worldSize;
        int workHeight = fragmentHeight + MASK_SIZE;
        
        uchar** imageArrays = matToArrays(inputImage);
        uchar** bluredArrays = new uchar*[chanelCount];
        for (int i = 0; i <chanelCount;i++)
        {  
            //OR uchar[inputImage.cols* inputImage.rows]
            bluredArrays[i] = new uchar[inputImage.cols +  inputImage.cols* inputImage.rows];
        }
        for(int j = 1; j< worldSize ; j++)
        {
            std::cout<<"sending image info to "<< j<<"\n";
            MPI_Send(&chanelCount, 1, MPI_INT, j, 0, MPI_COMM_WORLD);  
            MPI_Send(&imgWidth, 1, MPI_INT, j, 0, MPI_COMM_WORLD);            
            MPI_Send(&imgWidth, 1, MPI_INT, j, 0, MPI_COMM_WORLD);            
            MPI_Send(&workHeight, 1, MPI_INT, j, 0, MPI_COMM_WORLD);            
                    
            for (int i = 0; i <chanelCount;i++)
            {  
                std::cout<<"sending image data to "<< j<<" chanel: "<<i<<"\n";
                MPI_Send(&imageArrays[i][(j-1)*fragmentHeight], j == worldSize -1 ? fragmentHeight : workHeight, MPI_UNSIGNED_CHAR, j,0,MPI_COMM_WORLD);
                MPI_Recv(&bluredArrays[i][(j-1)*fragmentHeight], fragmentHeight, MPI_UNSIGNED_CHAR, j,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);            
            }
        }
    
        
        cv::Mat* bluredImage = arraysToMat(bluredArrays, chanelCount, inputImage.rows, inputImage.cols);      
        cv::imwrite(argv[2], *bluredImage);
    } 
    else
    {
	    int chanelCount;
        int imgWidth;
        int fragmentHeight;
        int workHeight;
        std::cout<<"getting image info at "<< worldRank <<"\n";
        MPI_Recv(&chanelCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout<<"got chanelCount at "<< worldRank <<"\n";
        MPI_Recv(&imgWidth, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&fragmentHeight, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&workHeight, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        uchar* buffer = new uchar[workHeight];
        
        for (int i = 0; i <chanelCount;i++)
        {  
            uchar* blured;
            std::cout<<"getting image data at "<< worldRank <<" chanel: "<<i<<"\n";
            MPI_Recv(buffer, workHeight, MPI_UNSIGNED_CHAR, 0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);  
            blured = Blur(buffer, imgWidth, fragmentHeight, workHeight);
            MPI_Send(blured, fragmentHeight, MPI_UNSIGNED_CHAR, 0,0,MPI_COMM_WORLD);     
        }
    }
    
    //Cleanup
    MPI_Finalize();
}
