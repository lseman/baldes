/*
   Author:  Jianfei Zhu  
            Concordia University
   Date:    Sep. 9, 2004

Copyright (c) 2004, Concordia University, Montreal, Canada
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   - Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
   - Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
   - Neither the name of Concordia University nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.

*/

/* This is an implementation of FP-growth* / FPmax* /FPclose algorithm.
 *
 * last updated Sep. 09, 2004
 *
 */
#include "fpmax.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <time.h>
#include "buffer.h"

#define LINT sizeof(int)

using namespace std;

FPmax* fpmax_inst;

void fpmax(char const * in, char const * out, unsigned int minsup)
{
	fpmax_inst = new FPmax(in, out, minsup);
	fpmax_inst->run();
	delete fpmax_inst;
}

FISet* fpmax(Dataset* dataset, unsigned int minsup, unsigned int nlargest)
{
	fpmax_inst = new FPmax(dataset, minsup, nlargest);
	FISet* frequentItemsets = fpmax_inst->run();
	delete fpmax_inst;
	
	return frequentItemsets;
}

FPmax::FPmax(char const * in, char const * out, int minsup) : fdat(new Data(in)), fout(new FSout(out)), minsup(minsup)
{
	if(!fdat->isOpen()) {
		cerr << in << " could not be opened!" << endl;
		exit(2);
	}
}

FPmax::FPmax(Dataset* dataset, int minsup, unsigned int nlargest) : fdat(new Data(dataset)), fout(new FSout(nlargest)), minsup(minsup)
{
}

FPmax::~FPmax()
{
	delete []ITlen;
	delete []bran;
	delete []prefix;
	delete []order_item;
	delete []item_order;
	if (current_fi != NULL) delete []current_fi;
	delete []compact;
	if (supp != NULL) delete []supp;
	delete list;
	delete fp_buf;
	delete fdat;
	delete fout;
}

void FPmax::printLen()
{
	int i, j, sum=0;
	for(i=ITEM_NO-1; i>=0&&ITlen[i]==0; i--);
        for(j=0; j<=i; j++)sum+=ITlen[j];
	printf("%d\n", sum);
	for(j=0; j<=i; j++) 
		printf("%d\n", ITlen[j]);
}

FISet* FPmax::run()
{
	THRESHOLD = minsup;
	
	int i;
	FI_tree* fptree;
	
	fp_buf=new memory(1000, 524288L, 1048576L, 2);
//	fp_buf=new memory(60, 4194304L, 8388608L, 2);
//	fp_buf=new memory(2000, 262144L, 524288L, 2);
	fptree = (FI_tree*)fp_buf->newbuf(1, sizeof(FI_tree));
	fptree->init(-1, 0);
	fptree -> scan1_DB(fdat);
	ITlen = new int[fptree->itemno];
	bran = new int[fptree->itemno];
	compact = new int[fptree->itemno];
	prefix = new int[fptree->itemno];

#ifdef CFI
		list=new stack(fptree->itemno, true); 
#else
		list=new stack(fptree->itemno); 
#endif

	assert(list!=NULL && bran!=NULL && compact!=NULL && ITlen!=NULL && prefix!=NULL);

	for(i =0; i < fptree->itemno; i++)
	{
		ITlen[i] = 0L;
		bran[i] = 0;
	}

	fptree->scan2_DB(fdat);
	// fdat->close();
	// if(fptree->itemno==0)return 0;

	// FSout* fout;
	// if(out)
	// {
		// fout = new FSout(/*out*/);

		//print the count of emptyset
#ifdef FI
		fout->printSet(0, NULL, TRANSACTION_NO);
#endif

#ifdef CFI
		if(TRANSACTION_NO != fptree->count[0])
			fout->printSet(0, NULL, TRANSACTION_NO);
#endif			
	// }else
		// fout = NULL;


	if(fptree->Single_path())
	{
		Fnode* node;
		int i=0;
		for(node=fptree->Root->leftchild; node!=NULL; node=node->leftchild)
		{
			list->FS[i++]=node->itemname;
#ifdef CFI
				list->counts[i-1] = node->count;
#endif
		}

#ifdef FI
			fptree->generate_all(fptree->itemno, fout);
#endif

#ifdef CFI
			int Count;
			i=0;
			while(i<fptree->itemno)
			{
				Count = list->counts[i];
				for(; i<fptree->itemno && list->counts[i]==Count; i++);
				ITlen[i-1]++;
				fout->printSet(i, list->FS, Count);
			}
#endif
		
#ifdef MFI
		if (fptree->itemno > 0)
		{
			fout->printSet(fptree->itemno, list->FS, fptree->head[fptree->itemno-1]->count);
			ITlen[i-1]=1;
		}
#endif
		// printLen();
		return fout->getFrequentItemsets();
	}

	current_fi = new bool[fptree->itemno];
	supp=new int[fptree->itemno];		//for keeping support of items
	assert(supp!=NULL&&current_fi!=NULL);

	for(i = 0; i<fptree->itemno; i++)
	{
		current_fi[i] = false;
		supp[i]=0;
	}

#ifdef MFI
	MFI_tree* LMFI;
		mfitrees = (MFI_tree**)new MFI_tree*[fptree->itemno];
	//	memory* Max_buf=new memory(40, 1048576L, 5242880, 2);
	        memory* Max_buf=new memory(250, 262144L, 524288L, 2);
		LMFI = (MFI_tree*)Max_buf->newbuf(1, sizeof(MFI_tree));
		LMFI->init(Max_buf, fptree, NULL, NULL, -1);
		fptree->set_max_tree(LMFI);
		mfitrees[0] = LMFI;
		fptree->FPmax(fout);
		fptree->free();
		delete Max_buf;
		delete []mfitrees;
#endif

#ifdef CFI
	CFI_tree* LClose;
		cfitrees = (CFI_tree**)new CFI_tree*[fptree->itemno];
//		memory* Close_buf=new memory(40, 1048576L, 5242880, 2);
	        memory* Close_buf=new memory(250, 262144L, 524288L, 2);
		LClose = (CFI_tree*)Close_buf->newbuf(1, sizeof(CFI_tree));
		LClose->init(Close_buf, fptree, NULL, NULL, -1);
		fptree->set_close_tree(LClose);
		cfitrees[0] = LClose;
		fptree->FPclose(fout);
		delete Close_buf;
		delete []cfitrees;
#endif

#ifdef FI
		fptree->FP_growth(fout);
#endif

    // printLen();
	// if(fout)
		// fout->close();

	// delete fp_buf;
	// delete list;
	// delete []current_fi;
	// delete []supp;
	// delete []ITlen;
	// delete []bran;
	// delete []compact;
	// delete []prefix;
	// delete []order_item;
	// delete []item_order;
							
	return fout->getFrequentItemsets();
}
