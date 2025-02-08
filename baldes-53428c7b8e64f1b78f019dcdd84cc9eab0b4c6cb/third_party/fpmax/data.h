/*----------------------------------------------------------------------
  File     : data.h
  Contents : data set management
----------------------------------------------------------------------*/
#ifndef _DATA_CLASS
#define _DATA_CLASS

#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <set>

#define TransLen 50

// Dataset is a type defined for passing in-memory input data (as a list of sets of integers)
typedef std::list<std::set<int>> Dataset;

class Transaction
{
public:
	
	Transaction():maxlength(TransLen), length(0){ t = new int[maxlength];}
	void DoubleTrans(int);				// if current item is greater than current longest transaction, we change the length of t as 2*item
	~Transaction(){delete []t;}
  
	int maxlength;
	int length;
	int *t;
};

class Data
{
 public:
	
	Data(char const *filename);
	Data(Dataset* dataset);
	~Data();
	int isOpen();
	void close(){if(in)fclose(in);}

	Transaction *getNextTransaction();
  
 private:
  
	FILE *in;
	Dataset* dataset;
	Dataset::iterator nextTransaction;
};

#endif

