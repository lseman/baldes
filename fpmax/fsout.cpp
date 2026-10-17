#include "fsout.h"
#include "common.h"

FSout::FSout(char const *filename) : frequentItemsets(NULL), nlargest(0)
{
  out = fopen(filename,"wt");
}

FSout::FSout(unsigned int nlargest) : out(NULL), frequentItemsets(new FISet), nlargest(nlargest)
{
}

FSout::~FSout()
{
	if(out)
	{
		fclose(out);
		delete frequentItemsets;
	}
}

int FSout::isOpen()
{
  if(out) return 1;
  else return 0;
}

void FSout::printSet(int length, int *iset, int support)
{
//#ifdef shown 
  if (out)
	  fprintf(out, "%d;%d;",length,support);
  FrequentItemset fi(support);
  for(int i=0; i<length; i++) 
  {
	if (out)
		fprintf(out, "%d ", fpmax_inst->order_item[iset[i]]);
	else if (frequentItemsets)
		fi.insert(fpmax_inst->order_item[iset[i]]);
//	printf("%d ", order_item[iset[i]]);
  }
  if (out)
	  fprintf(out,"\n");
  else if (frequentItemsets)
  {
	  frequentItemsets->insert(fi);
	  
	  if (nlargest && frequentItemsets->size() > nlargest)
		  frequentItemsets->erase(--frequentItemsets->end());
  }
  // fprintf(out, "(%d)\n", support);
//  printf("(%d)\n", support);
//#endif
}

void FSout::printset(int length, int *iset)	// Not used in FPmax*
{
//#ifdef shown 
  for(int i=0; i<length; i++) 
    fprintf(out, "%d ", fpmax_inst->order_item[iset[i]]);
//#endif
}

void FSout::close()
{
	if (out)
		fclose(out);
}

FISet* FSout::getFrequentItemsets()
{
	return frequentItemsets;
}

