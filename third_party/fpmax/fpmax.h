#include "fp_tree.h"

class FPmax
{
	public:
	
		FPmax(char const * in, char const * out, int minsup);
		FPmax(Dataset* dataset, int minsup, unsigned int nlargest);
		~FPmax();
	
		FISet* run();

		int *ITlen;
		int* bran;
		int* prefix;

		int* order_item;		// given order i, order_item[i] gives itemname
		int* item_order;		// given item i, item_order[i] gives its new order 
								//	order_item[item_order[i]]=i; item_order[order_item[i]]=i;
		bool* current_fi = NULL;
		int* compact;
		int* supp = NULL;

		MFI_tree** mfitrees;
		CFI_tree** cfitrees;

		stack* list;
		int TRANSACTION_NO=0;
		int ITEM_NO=100;
		int THRESHOLD;

		memory* fp_buf;
		
		void printLen();
	
	private:
		
		Data* fdat;
		FSout* fout;
		int minsup;
		unsigned int nlargest;
};

/*
Function for file-based input/output

Parameters:
- in: input file - must contain one transaction per line, with items represented as integer indices
- out: output file - the resulting frequent itemsets will be written to it (one per line, preceeded by its size and support)
- minsup: minimum (absolute) support of frequent itemsets

Returns void since the output is written to `out`
*/
void fpmax(char const * in, char const * out, unsigned int minsup);

/*
Function for in-memory input/output

Parameters:
- dataset: input data - each transaction is a set of integers and a dataset is a list of transactions (see `data.h` for the definition of the Dataset type)
- minsup: minimum (absolute) support of frequent itemsets
- nlargest: if > 0, only the <nlargest> largest frequent itemsets are returned

Returns a pointer to an FISet object - a set of FrequentItemset objects (see `fitemset.h` for their definitions)
*/
FISet* fpmax(Dataset* dataset, unsigned int minsup, unsigned int nlargest=0);
