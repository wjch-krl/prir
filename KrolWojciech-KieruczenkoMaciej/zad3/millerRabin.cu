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
    
    /**
    Check if number is propapbly prime
    **/
    virtual bool checkNumber(unsigned int number)
    {
        // return for small or even values
        if ( number==2 || number==3 ) 
        {
            return true;
        }
        if ( number<=1 || number % 2 == 0)
        {
            return false;
        }
        // Rewrite number-1 (this is event number) as multiper*2^exponent
        int exponent = 0;
        unsigned int tmp = number-1;
        while(!(tmp & 1))
        {
            exponent++;
            tmp >>= 1;
        }
        
        unsigned int multiper = (number-1) / (1<<exponent);
        
        //Alocate device memory
        curandState* devStates;
        unsigned int* devResult;
        unsigned int result;
        CHK_OK( cudaMalloc ( &devStates, numberOfIterations*sizeof( curandState ) ));
        CHK_OK( cudaMalloc ( &devResult, sizeof(unsigned int) )); 
        //Set flag to 0
        CHK_OK( cudaMemset ( devResult, 0, sizeof(unsigned int)));  
        //Init curand
        initCurand<<<(numberOfIterations+31)/32,32>>>(devStates,123,numberOfIterations);
        //Check for prime number
        checkPrimeMillerRabin<<<(numberOfIterations+31)/32,32>>>(devStates,number,multiper,numberOfIterations,exponent,devResult);
        //Copy result back to Host
        CHK_OK(cudaMemcpy(&result, devResult, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        return result == 0;
    }

};