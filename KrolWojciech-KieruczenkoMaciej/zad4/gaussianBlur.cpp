#include <iostream>
#include <mpi.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define MASK_SIZE 5

int weights[MASK_SIZE][MASK_SIZE] = {
    {2,2,4,2,2},
    {2,4,8,4,2},
    {4,8,16,8,4},
    {2,4,8,4,2},
    {2,2,4,2,2},
};;

uchar* Blur(uchar* image, int width, int startRow, int endRow)
{
    std::cout<<"got image imgWidth:"<<width<<" startRow:"<<startRow<<" endRow:"<<endRow<<"\n";
    uchar* resultImage = new uchar[width*(endRow - startRow)];
    int offset = MASK_SIZE / 2;
    for(int i= 0; i< width; i++)
    {
        for (int j = startRow; j< endRow; j++)
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
                    if(k<width)
                    {
                        pixelValue += weights[weightsIdx][weightsIdy]*image[k+width*l];
                        weightsSum += weights[weightsIdx][weightsIdy];
                    }
                }
            }
            if(weightsSum > 0)
            {
                pixelValue /= weightsSum;
            }
            if (pixelValue > 255)
            {
                pixelValue = 255;
            }
            else if (pixelValue < 0)
            {
                pixelValue = 0;
            }
            resultImage[i+width*(j-startRow)] = (uchar)pixelValue;
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

void getOffset(int machineCount,int machineNumber, int* upper, int* lower)
{
    if(machineNumber == 0)
    {
        *upper =0;
    } 
    else 
    {
        *upper = MASK_SIZE / 2;        
    } 
    if (machineNumber == machineCount -1)
    {
        *lower =0;
    }
    else
    {
        *lower = MASK_SIZE / 2;
    }
}

int main(int argc, char** argv)
{
    int worldSize;
    int worldRank;
    
    if(argc != 3)
    {
        std::cout<<"Usage: \nmpirun "<<argv[0]<< " INPUT_FILE_PATH OUTPUT_FILE_PATH \n";
        exit(EXIT_FAILURE);
    }
    //GaussianBlur* gb = new GaussianBlur();
    // Initialize MPI
    MPI_Init(NULL, NULL);

    // Get number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Get process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Request* asyncRequests = new MPI_Request[worldSize - 1];

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

        uchar** imageArrays = matToArrays(inputImage);
        uchar** bluredArrays = new uchar*[chanelCount];
        for (int i = 0; i <chanelCount;i++)
        {  
            //OR uchar[inputImage.cols* inputImage.rows]
            bluredArrays[i] = new uchar[inputImage.cols +  inputImage.cols* inputImage.rows];
        }
        
        int splitCount = worldSize - 1;
        for(int j = 0; j< splitCount; j++)
        {
            MPI_Request req;
            int targetId = j + 1;
            MPI_Isend(&chanelCount, 1, MPI_INT, targetId, 0, MPI_COMM_WORLD,&req);  
            MPI_Request_free(&req);
            MPI_Isend(&imgWidth, 1, MPI_INT, targetId, 0, MPI_COMM_WORLD,&req);      
            MPI_Request_free(&req);
            MPI_Isend(&imgHeight, 1, MPI_INT, targetId, 0, MPI_COMM_WORLD,&req);            
            MPI_Request_free(&req);
            

                    
            for (int i = 0; i <chanelCount;i++)
            {  
                int startRow;
                int endRow;
                getOffset(splitCount, j, &startRow,&endRow);
                int end = imgHeight / splitCount * (j+1) + endRow;
                int start = imgHeight / splitCount * (j) - startRow;
                int count = end - start;
                std::cout<<"end "<< end << " start " << start << " count " << count <<"\n";                
                MPI_Isend(&imageArrays[i][start*imgWidth], count*imgWidth, MPI_UNSIGNED_CHAR, 
                    targetId,0,MPI_COMM_WORLD,&req);
                MPI_Request_free(&req);
                MPI_Irecv(&bluredArrays[i][(imgHeight / splitCount * (j))*imgWidth], imgHeight/splitCount*imgWidth,
                     MPI_UNSIGNED_CHAR, targetId,0,MPI_COMM_WORLD, &asyncRequests[j]); 
                std::cout<<"got blured \n";
            }
        }
        //Wait for all taks to complete
        for(int j = 0; j< worldSize -1 ; j++)
        {
            MPI_Wait(&asyncRequests[j], NULL);
        }
        cv::Mat* bluredImage = arraysToMat(bluredArrays, chanelCount, inputImage.rows, inputImage.cols);      
        cv::imwrite(argv[2], *bluredImage);
    } 
    else
    {
	    int chanelCount;
        int imgWidth;
        int imgHeight;
        int bufferSize;
        MPI_Recv(&chanelCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&imgWidth, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&imgHeight, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int fragmentCount = worldSize - 1;
        
        bufferSize = (imgHeight / fragmentCount + MASK_SIZE)*imgWidth;
        uchar* buffer = new uchar[bufferSize];
       
        int startRow;
        int endRow;
        getOffset(fragmentCount, worldRank,&startRow,&endRow);
        // int end = imgHeight / worldSize * (worldRank+1) + endRow;
        // int start = imgHeight / worldSize * (worldRank) - startRow;
        int count = imgHeight / fragmentCount - endRow - startRow;
                
        for (int i = 0; i <chanelCount;i++)
        {  
            uchar* blured;
           // std::cout<<"end "<< end << " start " << start << " count " << count <<"\n";
            MPI_Recv(buffer, bufferSize, MPI_UNSIGNED_CHAR, 0,0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
            blured = Blur(buffer, imgWidth, startRow, imgHeight / fragmentCount); //dodac wysokość
            std::cout<<"sending blured image data at "<< worldRank <<" chanel: "<<i<<"\n";
            MPI_Send(blured, imgWidth*(imgHeight / fragmentCount - startRow), MPI_UNSIGNED_CHAR, 0,0,MPI_COMM_WORLD);     
        }
    }
    
    //Cleanup
    MPI_Finalize();
}
