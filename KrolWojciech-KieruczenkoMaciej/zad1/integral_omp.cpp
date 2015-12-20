#include <cstdlib>
#include <iostream>
#include <chrono>
#include "exprtk.hpp"

using namespace std;

///Abstract class representing parser
class IParser
{
public:
    virtual double
    Eval (double x) = 0;
};

///Abstract class thar defines method for integral calculation
class IIntegral
{
public:
    virtual double
    Calculate (double from, double to, int nSteps) = 0;
};

///Abstract class that defines parser factory method
class IParserFactory{
public:
    virtual IParser*
    Build () = 0;
};

///Class that implements method to calculate integral using omp parallel for
class OmpIntegral : IIntegral
{
public:
    
    OmpIntegral (int nThreads, IParserFactory* parserFactory)
    {
        this->parserFactory = parserFactory;
        this->nThreads = nThreads;
    }
    
    double
    Calculate (double from, double to, int nSteps)
    {
        double step = fabs (from - to) / nSteps;
        double result = 0;
        double x;
        double tmp;
        //private x, tmp - variables which thread saves calculates values
        //from,to,nSteps - read only variables can be shared
#pragma omp parallel num_threads(nThreads) private(x,tmp) shared(from,to,nSteps)
        {
            //One Parser per thread
            IParser* parser = parserFactory->Build();
            //Caculates integral using following exuation
            //(a1+a2)/2*h + (a2+a3)/2*h + .. + (an-1+an)/2*h = [(a1+an)/2 + a2+a3+...an-1]*h
            //Reduction (sum) on result
#pragma omp for reduction(+:result)
            for (int i = 1; i < nSteps -1; i++)
            {
                x = from + fabs(from - to) * (i / (double)nSteps);
                tmp = parser->Eval (x);
                result += tmp;
            }
#pragma omp single
            {
                result += (parser->Eval (from) + parser->Eval(to)) / 2;
            }
        }
        return result*step;
    }
    
private:
    IParserFactory* parserFactory;
    int nThreads;
};

///Fixed function i IParser implementation used for testing
class SimpleParser : public IParser
{
public:
    
    virtual double
    Eval (double x)
    {
        return sin (2 * x) + 2 * x;
    }
};

///IParser implementation that utilizes exprtk.hpp lib
class ExprtkParser : public IParser
{
public:
    
    ExprtkParser (string& expression)
    {
        exprtk::symbol_table<double> symbol_table;
        symbol_table.add_constants ();
        symbol_table.add_variable ("x", this->x);
        expr.register_symbol_table (symbol_table);
        exprtk::parser<double> parser;
        
        if (!parser.compile (expression, expr))
        {
            throw parser.error ().c_str ();
        }
    }
    
    virtual double
    Eval (double x)
    {
        this->x = x;
        return this->expr.value ();
    }
private:
    exprtk::expression<double> expr;
    double x;
};

///Parser factory
class ParserFactory : public IParserFactory{
public:
    ParserFactory(string expression)
    {
        this->expression = expression;
    }
    
    virtual IParser* Build (){
        return new ExprtkParser(expression);
    }
private:
    string expression;
};

int
main (int argc, char** argv)
{
    int threadCount;
    double start;
    double stop ;
    int count;
    string expr;
    if(argc != 6)
    {
        cerr<<"Too many/less parameters.\n";
        return EXIT_FAILURE;
    }
    try
    {
        threadCount = stoi (argv[1]);
        start = stod (argv[2]);
        stop = stod (argv[3]);
        count = stoi (argv[4]);
        expr = argv[5];
    }
    catch (invalid_argument)
    {
        cerr<<"Invalid parameters.\n";
        return EXIT_FAILURE;
    }
    catch (out_of_range)
    {
        cerr<<"Parameters out of range.\n";
        return EXIT_FAILURE;
    }
    if(start> stop)
    {
        cerr<<"End value must be greater then start value.\n";
        return EXIT_FAILURE;
    }
    if(threadCount<1 || count <1)
    {
        cerr<<"Count must be > 0.\n";
        return EXIT_FAILURE;
    }
    IParserFactory* parser = new ParserFactory (expr);
    OmpIntegral* integral = new OmpIntegral (threadCount, parser);
    
    auto t1 = chrono::high_resolution_clock::now();
    double value = integral->Calculate (start, stop, count);
    auto t2 = chrono::high_resolution_clock::now();
    
    cout << "Time: " <<  chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << " ms\n";
    cout << "Integral value: " << value << "\n";
    return EXIT_SUCCESS;
}

