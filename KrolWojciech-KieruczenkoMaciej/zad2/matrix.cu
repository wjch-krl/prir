#include "errors.hu" 
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include "kernel.cu"

class Matrix
{
private:
    double *hostMtrx;
    double *deviceMtrx;
    unsigned int mtrxSize;
    bool isDecomposed;
    int blockCount;
    int threadCount;
    /*
    Copy entire matrix to device
     */
    inline void copyToDevice()
    {
        CHK_OK(cudaMemcpy(deviceMtrx, hostMtrx, mtrxSize * mtrxSize * sizeof (double), cudaMemcpyHostToDevice));
    }

    /*
    Copy entire matrix to host
     */
    inline void copyToHost()
    {
        CHK_OK(cudaMemcpy(hostMtrx, deviceMtrx, mtrxSize * mtrxSize * sizeof (double), cudaMemcpyDeviceToHost));
    }

    /*
    Copy count elements of matrix starting from startIdx (to device)
     */
    inline void copyToDevice(int startIdx, int count)
    {
        CHK_OK(cudaMemcpy(&deviceMtrx[startIdx], &hostMtrx[startIdx], count * sizeof (double), cudaMemcpyHostToDevice));
    }

    /*
    Copy count elements of matrix starting from startIdx (to host)
     */
    inline void copyToHost(int startIdx, int count)
    {
        CHK_OK(cudaMemcpy(&hostMtrx[startIdx], &deviceMtrx[startIdx], count * sizeof (double), cudaMemcpyDeviceToHost));
    }


public:

    /*
    Alocate memory (both host and device) and load matrix from file
     */
    Matrix(std::string path,int blockCount, int threadCount) : isDecomposed(false)
    {
        this->blockCount = blockCount;
        this->threadCount = threadCount;
        size_t mtrxSize;
        int elementsCount;
        this->mtrxSize = 1;
        std::ifstream in(path.c_str());

        std::string firstLine; // Used to determine size of the matrix in file
        if (!in)
        {
            std::cout << "Cannot open file. \n";
            exit(1);
        }
        std::getline(in, firstLine);
        /* Count elements in first row */
        for (int i = 0; i < firstLine.size(); ++i)
        {
            if (firstLine[i] == ' ')
            {
                ++this->mtrxSize;
            }
        }
        elementsCount = this->mtrxSize * this->mtrxSize;
        mtrxSize = elementsCount * sizeof (double);
        /*Allocate host memory*/
        hostMtrx = (double *) malloc(mtrxSize);
        if (!hostMtrx)
        {
            std::cerr << "Host alocation error \n";
        }
        /* Reset file reader*/
        in.close();
        in.clear();
        in.open(path.c_str());
        /*Read matrix*/
        for (int i = 0; i < elementsCount; i++)
        {
            in >> hostMtrx[i];
        }
        in.close();
        /* Alock and copy matrix to device */
        CHK_OK(cudaMalloc((void **) &deviceMtrx, mtrxSize))
        copyToDevice();
    }

    /*
    Free allocated memory
     */
    ~Matrix()
    {
        if (hostMtrx)
        {
            free(hostMtrx);
            hostMtrx = NULL;
        }
        if (deviceMtrx)
        {
            cudaFree(deviceMtrx);
            deviceMtrx = NULL;
        }
    }

    void decomposeGPU()
    {
        int idx = 0;
        //foreach row
        while (idx < mtrxSize)
        {
            //Calculate indexes
            int startIdx = (idx * mtrxSize + idx);
            int endIdx = (idx * mtrxSize + mtrxSize);
            //First decomposition step
            for (int j = startIdx + 1; j < endIdx; j++)
            {
                hostMtrx[j] = (hostMtrx[j] / hostMtrx[startIdx]);
            }
            //Update device matrix
            copyToDevice(startIdx, mtrxSize - idx);
            //Second decomposition step - on GPU
            //decompositionKernel <<<(mtrxSize - idx - 1), 1 >>>(deviceMtrx, mtrxSize, idx);
            decompositionKernel <<<blockCount, threadCount>>>(deviceMtrx, mtrxSize, idx,mtrxSize - idx - 1);
            
            idx++;
            // if(idx<mtrxSize)
            // {
            //if not last iteration 
            //Update host matrix
            startIdx = (idx * mtrxSize + idx);
            copyToHost(startIdx, mtrxSize - idx);
            //}
        }
        this->isDecomposed = true;
    }

    /*
    Decomposition on CPU using Doolite alghotim based on paper:
    http://www.engr.colostate.edu/~thompson/hPage/CourseMat/Tutorials/CompMethods/doolittle.pdf
     */
    void decomposeCPU()
    {
        int i, j, k;
        for (i = 0; i < mtrxSize; i++)
        {
            for (j = i; j < mtrxSize; j++)
            {
                for (k = 0; k < i; k++)
                {
                    hostMtrx[i * mtrxSize + j] -= hostMtrx[i * mtrxSize + k] * hostMtrx[k * mtrxSize + j];
                }
            }
            for (j = i + 1; j < mtrxSize; j++)
            {
                for (k = 0; k < i; k++)
                {
                    hostMtrx[j * mtrxSize + i] -= hostMtrx[j * mtrxSize + k] * hostMtrx[k * mtrxSize + i];
                }
                hostMtrx[j * mtrxSize + i] /= hostMtrx[i * mtrxSize + i];
            }
        }
        copyToDevice();
        this->isDecomposed = true;
    }

    /*
    Computes determinant of matrix. (Decomposes matrix, if if havn't been done yet)
     */
    long double determinant()
    {
        if (!this->isDecomposed)
        {
            this->decomposeGPU();
        }
        double det = 1;
        /*
        Determinant of LU decomposed matrix = sum of diagonal elements from U matrix
         */
        for (int i = 0; i < mtrxSize; i++)
        {
            det *= hostMtrx[i * (mtrxSize + 1)];
        }
        return det;
    }
};