#include "common.hu"
#define NUMBERS_IN_CELL (sizeof(int)*8)

class EratosthenesSievePrimeChecker : public PrimeChecker 
{
private:
    int* hostArray;
    int* deviceArray;
    unsigned int maxValue;
    unsigned int arraySize;
    
    inline void getIdxAndMask(unsigned int number, int* idx, int* mask)
    {
        *idx = number / NUMBERS_IN_CELL;
        *mask = (1 <<(number % NUMBERS_IN_CELL));
    }
    
    inline void copyDtoH()
	{
		CHK_OK(cudaMemcpy(hostArray, deviceArray, arraySize*sizeof(int), cudaMemcpyDeviceToHost));
	}
    
     void generatePrimesGPU()
    {
        CHK_OK(cudaMalloc((void **)&deviceArray, arraySize*sizeof(int)))
        CHK_OK(cudaMemset(deviceArray, 0, arraySize*sizeof(int)))
        int gridSize = maxValue < 32 ? 1 : maxValue / 32;
        //int iterations = 0;
        if(gridSize > 65535)
        {
            gridSize = 65535;
        }
        gridSize = 10;
        dim3 grids(gridSize, gridSize);
        dim3 threads(32, 32);
        detatchNumber<<<grids,threads>>>(deviceArray,maxValue);
        copyDtoH();
        CHK_OK(cudaFree(deviceArray))        
    }

    void generatePrimesCPU()
    {
        double maxToCheck = ceil(sqrt(maxValue));
        int idx, mask;
        for(int number=2; number<=maxToCheck;number++)
        {
            for(int currentNumber=number; currentNumber < maxValue; currentNumber+=number)
            {
                getIdxAndMask(currentNumber,&idx,&mask);
                hostArray[idx] |= mask;
            }
        }
    }
    
public:
    EratosthenesSievePrimeChecker(unsigned int maxValue)
    {
        this->maxValue = maxValue;
        this->arraySize = maxValue / NUMBERS_IN_CELL;
        if (maxValue % NUMBERS_IN_CELL != 0)
        {
            arraySize++;
        }
        this->hostArray = new int[arraySize];
        generatePrimesGPU();
    }
    
    virtual bool checkNumber(unsigned int number)
    {
        int idx;
        int mask;
        getIdxAndMask(number,&idx,&mask);
        int tested =hostArray[idx];
        return !(tested & mask);
    }
};