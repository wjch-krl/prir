#include "common.hu"

#define NUMBERS_IN_CELL (sizeof(int)*8)

__device__ void getIdxAndMask(unsigned int number, int* idx, int* mask)
{
    *idx = number / NUMBERS_IN_CELL;
    *mask = (1 <<(number % NUMBERS_IN_CELL));
}

__global__ void setupRandom( curandState * state, unsigned long seed, int k)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while(idx<k)
    {
        curand_init (seed, idx, 0, &state[idx]);
        idx+=blockDim.x * gridDim.x;  
    }
} 

__device__ void rand( curandState* globalState, unsigned int min,unsigned int max, unsigned int* rand) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = globalState[idx];
    float randomFloat = curand_uniform( &localState );
    randomFloat *= (max - min + 0.999999);
    randomFloat += min;
    *rand = (unsigned int)randomFloat;
    globalState[idx] = localState; 
}

__global__ void detatchNumber(int* deviceArray, unsigned int maxValue)
{
    int id, mask;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if(idx > 1 && idy > 1 )
    {
        unsigned long number = idx * idy;
        while(number <= maxValue)
        {
            // getIdxAndMask(number,&id, &mask);
            // atomicOr(&deviceArray[id], mask);
            int newIdy = idy;
            while(number <= maxValue)
            {
                getIdxAndMask(number,&id, &mask);
                atomicOr(&deviceArray[id], mask);
    
                newIdy += blockDim.y * gridDim.y;   
                number = idx * newIdy;          
            }
            idx += blockDim.x * gridDim.x;         
            number = idx * idy;          
        }
    }
}

/**
Function for calculation a^x mod n, based on
http://en.wikipedia.org/wiki/Modular_exponentiation
**/
__device__ void powerModulo(unsigned int a, unsigned int x, unsigned int n, unsigned int* result)
{
    unsigned long rLong=1;
    unsigned long aLong=a;
    while (x != 0) 
    {
        if ((x & 1) == 1)
        {
            rLong = aLong*rLong % n;
        }
        aLong = aLong*aLong % n;
        x >>= 1;
    }
    *result = (unsigned int)rLong;
}



__device__ void checkPrimeNumber(curandState* globalState, unsigned int n,unsigned int d,int s,unsigned int* result)
{
    unsigned int a;
    rand(globalState,2, n-2,&a);
    unsigned int x;
    powerModulo(a,d,n,&x);

    if ( x == 1 || x == n-1 )
    {
        return;
    }
    for ( int r = 1; r <= s-1; ++r ) 
    {
        powerModulo(x, 2, n, &x);
        if ( x == 1 ) 
        {
            atomicAdd(result,1);
            return;
        }
        if ( x == n - 1 )
        {
            return;
        }
        if(*result != 0)
        {
            return;
        }
    }
    atomicAdd(result,1);
}

__global__ void checkPrimeMillerRabin( curandState* globalState, unsigned int n,unsigned int d, int k, int s,unsigned int* result)
{
     int idx = threadIdx.x + blockIdx.x * blockDim.x;
     while(idx<k)
     {
        checkPrimeNumber(globalState,n,d,s,result);
        if(*result != 0)
        {
            return;
        }
        idx+=blockDim.x * gridDim.x;  
    }
}
