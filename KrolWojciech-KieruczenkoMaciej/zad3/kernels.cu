#include "common.hu"

#define NUMBERS_IN_CELL (sizeof(int)*8)

/**
Initializes Curand lib
**/
__global__ void initCurand( curandState * state, unsigned long seed, int count)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while(idx<count)
    {
        //inicjalizacja bibloteki curand - dla każdego indexu
        curand_init (seed, idx, 0, &state[idx]);
        idx+=blockDim.x * gridDim.x;  
    }
} 

/**
Generates random integer between given values
**/
__device__ void rand( curandState* globalState, unsigned int min,unsigned int max, 
    unsigned int* rand) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = globalState[idx];
    //losowanie liczby zmienno przecinkowej z przediału 0 .. 1
    float randomFloat = curand_uniform( &localState );
    //Konwersja do liczby całkowitej w zadanym przedziale
    randomFloat *= (max - min + 0.999999);
    randomFloat += min;
    *rand = (unsigned int)randomFloat;
    globalState[idx] = localState; 
}

/**
Function for calculation a^x mod number, based on
http://en.wikipedia.org/wiki/Modular_exponentiation
**/
__device__ void powerModulo(unsigned int a, unsigned int x, unsigned int number, 
    unsigned int* result)
{
    unsigned long rLong=1;
    unsigned long aLong=a;
    while (x != 0) 
    {
        if ((x & 1) == 1)
        {
            rLong = aLong*rLong % number;
        }
        x >>= 1;
        aLong = aLong*aLong % number;
    }
    *result = (unsigned int)rLong;
}

/**
Checks given number if prime (single iteration of Miller-Rabin algorithm)
**/
__device__ void checkPrimeNumber(curandState* globalState, unsigned int number,
    unsigned int exponent,int multiper,unsigned int* result)
{
    unsigned int a;
    //Get random int between 2 and number - 2
    rand(globalState,2, number-2,&a);
    unsigned int x;
    powerModulo(a,exponent,number,&x);
    if ( x == 1 || x == number-1 )
    {
        return;
    }
    for ( int i = 1; i <= multiper-1; i++ ) 
    {
        powerModulo(x, 2, number, &x);
        if ( x == 1 ) 
        {
            //Not a prime - return and set flag
            atomicAdd(result,1);
            return;
        }
        if ( x == number - 1 )
        {
            return;
        }
        //Check for other threads result.
        //Check every 10th iteration to avoid slow global memory acces
        if(i % 10 == 0 && *result != 0)
        {
            return;
        }
    }
    //Not a prime - set flag
    atomicAdd(result,1);
}

__global__ void checkPrimeMillerRabin( curandState* globalState, unsigned int number,
    unsigned int exponent, int count, int multiper,unsigned int* result)
{
     int idx = threadIdx.x + blockIdx.x * blockDim.x;
     //If indx < count (precision) then Chceck number     
     while(idx<count)
     {
        checkPrimeNumber(globalState,number,exponent,multiper,result);
        if(*result != 0)
        {
            return;
        }
        idx+=blockDim.x * gridDim.x;  
    }
}
