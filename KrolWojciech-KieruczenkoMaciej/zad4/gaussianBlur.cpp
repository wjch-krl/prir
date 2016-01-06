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
    std::cout<<"create buffer for"<<endRow - startRow<<" rows.\n";
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

void masterTask(int worldSize, const char* inputPath, const char* resultPath,MPI_Request* asyncRequests)
{
    cv::Mat inputImage = cv::imread(inputPath, 1 );
    if(! inputImage.data )                             
    {
        throw std::string("Invalid image \n");
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
            if(j == worldSize - 1)
            {
                count += imgHeight % splitCount;
            }
            std::cout<<"Master: Sending " << count <<" rows \n";                
            MPI_Isend(&imageArrays[i][start*imgWidth], count*imgWidth, MPI_UNSIGNED_CHAR, 
                targetId,0,MPI_COMM_WORLD,&req);
            MPI_Request_free(&req);
            MPI_Irecv(&bluredArrays[i][(imgHeight / splitCount * (j))*imgWidth], (imgHeight/splitCount)*imgWidth,
                    MPI_UNSIGNED_CHAR, targetId,0,MPI_COMM_WORLD, &asyncRequests[j]); 
        }
    }
    //Wait for all taks to complete
    for(int j = 0; j< worldSize -1 ; j++)
    {
        MPI_Wait(&asyncRequests[j], NULL);
    }
    cv::Mat* bluredImage = arraysToMat(bluredArrays, chanelCount, inputImage.rows, inputImage.cols);      
    cv::imwrite(resultPath, *bluredImage);
}

void slaveTask(int worldSize, int worldRank)
{
    int chanelCount;
    int imgWidth;
    int imgHeight;
    int bufferSize;
    int startRow;
    int endRow;
    int count;    
    uchar* buffer;
    int fragmentCount;
    
    MPI_Recv(&chanelCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&imgWidth, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&imgHeight, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    fragmentCount = worldSize - 1;
    
    bufferSize = (imgHeight / fragmentCount + MASK_SIZE)*imgWidth;
    buffer = new uchar[bufferSize];

    getOffset(fragmentCount, worldRank,&startRow,&endRow);
    // int end = imgHeight / worldSize * (worldRank+1) + endRow;
    // int start = imgHeight / worldSize * (worldRank) - startRow;
    count = imgHeight / fragmentCount - endRow - startRow;
    if(worldRank == worldSize - 1)
    {
        count += imgHeight % fragmentCount;
    }
    for (int i = 0; i <chanelCount;i++)
    {  
        uchar* blured;
        // std::cout<<"end "<< end << " start " << start << " count " << count <<"\n";
        MPI_Recv(buffer, bufferSize, MPI_UNSIGNED_CHAR, 0,0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
        blured = Blur(buffer, imgWidth, startRow, imgHeight / fragmentCount +MASK_SIZE/2); //dodac wysokość
        std::cout<<"Slave: Sending "<< imgHeight / fragmentCount - startRow <<" rows \n";
        MPI_Send(blured, imgWidth*(imgHeight / fragmentCount - startRow +MASK_SIZE/2), MPI_UNSIGNED_CHAR, 0,0,MPI_COMM_WORLD);     
    }
}

int main(int argc, char** argv)
{
    int worldSize;
    int worldRank;
    
    if(argc != 3)
    {
        std::cerr<<"Usage: \nmpirun "<<argv[0]<< " INPUT_FILE_PATH OUTPUT_FILE_PATH \n";
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
        try
        {
            masterTask(worldSize, argv[1],argv[2],asyncRequests);
        } 
        catch(std::string e)
        {
            std::cerr << e;            
            exitFailure();            
        } 
        catch(...)
        {
            exitFailure();
        }
    } 
    else
    {
        try
        {
	       slaveTask(worldSize,worldRank);
        } 
        catch(std::string e)
        {
            std::cerr << e;            
            exitFailure();
        } 
        catch(...)
        {
            exitFailure();
        }
    }
    //Cleanup
    MPI_Finalize();
}

void exitFailure()
{
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);    
    MPI_Finalize();        
    exit(EXIT_FAILURE);
}