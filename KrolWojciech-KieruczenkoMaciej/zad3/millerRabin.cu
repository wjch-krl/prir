#include "common.hu"

class MillerRabinPrimeChecker : public PrimeChecker
{
private:
    unsigned int numberOfIterations;
    
public:
    MillerRabinPrimeChecker(unsigned int numberOfIterations)
    {
        this->numberOfIterations = numberOfIterations;
    }
    
    virtual bool checkNumber(unsigned int n)
    {
        // Must have ODD n greater than THREE
        if ( n==2 || n==3 ) return true;
        if ( n<=1 || n % 2 == 0) return false;
        
        // Write n-1 as d*2^s by factoring powers of 2 from n-1
        int s = 0;
        for ( unsigned int m = n-1; !(m & 1); ++s, m >>= 1 ); 
        
        unsigned int d = (n-1) / (1<<s);
        
        curandState* devStates;
        unsigned int* devResult;
        unsigned int result;
        CHK_OK( cudaMalloc ( &devStates, numberOfIterations*sizeof( curandState ) ));
        CHK_OK( cudaMalloc ( &devResult, sizeof(unsigned int) )); 
        CHK_OK( cudaMemset( devResult, 0, sizeof(unsigned int)));  
            
        setupRandom<<<(numberOfIterations+31)/32,32>>>(devStates,123,numberOfIterations);
        checkPrimeMillerRabin<<<(numberOfIterations+31)/32,32>>>(devStates,n,d,numberOfIterations,s,devResult);
        
        CHK_OK(cudaMemcpy(&result, devResult, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        // n is *probably* prime
        return result == 0;
    }

};