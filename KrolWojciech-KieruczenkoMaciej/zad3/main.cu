#include <tuple>
#include "common.hu"
#include "millerRabin.cu"

void processFile(std::string &filePath, std::vector<unsigned int> *numbers,unsigned int *max)
{
    unsigned int value;
    std::ifstream in(filePath.c_str());
    unsigned int maxTmp = 0;
    /*Read numbers*/
    while(in)
    {
        in >> value;
        //Find max element in file
        maxTmp = value > maxTmp ? value : maxTmp;
        numbers->push_back(value);
    }
    *max = maxTmp;
}

std::vector<std::tuple<unsigned int, bool>>* check(std::vector<unsigned int> *numbers,PrimeChecker* checker)
{    
    std::vector<std::tuple<unsigned int, bool>> *result = 
        new std::vector<std::tuple<unsigned int, bool>>(numbers->size());
    int i = 0;
    for (auto &number : *numbers) // access by reference to avoid copying
    {  
        result->at(i++) = std::make_tuple(number,checker->checkNumber(number));
    }
    return result;
}

void printResult(std::vector<std::tuple<unsigned int, bool>>* result)
{
    for (auto &number : *result) 
    {
        std::cout << std::get<0>(number) << ": ";
        if(std::get<1>(number))
        {
            std::cout<< "prime";
        } 
        else
        {
            std::cout<< "composite";
        } 
        std::cout << "\n";   
    } 
}


int main(int argc, char** argv)
{
	std::string path;
	if(argc != 2)
	{
		/* Print usage */
		std::cout<< "USSAGE:\n "<< argv[0] <<" FILE_PATH\n";
		return EXIT_FAILURE;
	}
	path = argv[1];

    std::vector<unsigned int> *numbers = new std::vector<unsigned int>();
    unsigned int max;
    processFile(path,numbers,&max);
    PrimeChecker *checker = new MillerRabinPrimeChecker(100);
    
	/*
	Time mesurment using std::chrono.
	I personally dont think that in this case using cuda events will be better,
	becouse we are mesuring time of entire process which is part only accelated using GPU 
	(Some important calculation are made on CPU)
	*/
	auto t1 = std::chrono::high_resolution_clock::now();
	/* Calculate determinant, next print value and computation time */
    auto result = check(numbers,checker);
	auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " <<  std::chrono::duration_cast<std::chrono::milliseconds>(t2 
		- t1).count() << " ms\n";
    //Print result
    printResult(result);   
    delete checker;
    return 0;
}

