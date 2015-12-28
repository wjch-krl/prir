#include <cstdio>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <mpi.h>

cv::Mat* splitImage(cv::Mat originalImage,int splitsCount)
{
    cv::Size imgSize;
    int splitWidth;
    cv::Mat* result;
    result = new cv::Mat[splitsCount];
    imgSize = originalImage.size();
    splitWidth = imgSize.width / splitsCount;
    for(int i=0;i<splitsCount;i++)
    {
        cv::Mat tmp;
        tmp = cv::Mat(originalImage, cv::Rect(i*splitWidth, 0, (i+1)*splitWidth, imgSize.height));
        result[i] = tmp;
    }
    return result;
}



int main(int argc, char** argv)
{
    int worldSize;  
    int worldRank;
    cv::Mat inputImage;
    
    if(argc != 3)
    {
        std::cout<<"USAGE: "<< argv[0]<<" input.jpg output.jpg\n";
    }
    
    //Init MPI communicator, get number of processes and current rank
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
 
    if(worldRank == 0)
    {
        cv::Mat* splittedImage;
        inputImage = cv::imread(argv[1], 1 );
        if(!inputImage.data)
        {
            std::cout<< "Invalid input image.\n";
            exit(EXIT_FAILURE);
        }
        splittedImage = splitImage(inputImage,worldSize);
        std::cout<<"\n\n"<<splittedImage[0]<<"\n\n";
    }
    
    //MPI Cleanup
    MPI_Finalize();
}
