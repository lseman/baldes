#ifndef _FSOUT_CLASS
#define _FSOUT_CLASS

#include <stdio.h>
#include "fitemset.h"

class FSout
{
 public:

  FSout(char const *filename);
  FSout(unsigned int nlargest);
  ~FSout();

  int isOpen();

  void printset(int length, int *iset);
  void printSet(int length, int *iset, int support);
  void close();
  FISet* getFrequentItemsets();

 private:

  FILE *out;
  FISet* frequentItemsets;
  unsigned int nlargest;
};

#endif

