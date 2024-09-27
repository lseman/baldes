/*
   Author:  Jianfei Zhu  
            Concordia University
   Date:    Sep. 26, 2003

Copyright (c) 2003, Concordia University, Montreal, Canada
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

#ifndef _FP_NODE_CLASS
#define _FP_NODE_CLASS

//class FP_tree;
class FI_tree;
class MFI_tree;
class CFI_tree;

class FPnode{
public:
	int itemname;

public:
//	FPnode(FPnode* par, int itemname, int count);
//	~Fnode();
};

class Fnode : public FPnode{
public:
	Fnode* par;
	Fnode* leftchild;
	Fnode* rightsibling;
	Fnode* next;
	int count;

public:
	Fnode(Fnode* par, int itemname, int count);
	~Fnode();

	void init(Fnode*, int, int);
    Fnode* append(FI_tree*, Fnode*, int, int);
};   

class Mnode : public FPnode{
public:
	Mnode* par;
	Mnode* leftchild;
	Mnode* rightsibling;
	Mnode* next;
	int level;      //level in MFI-tree

public:
	~Mnode();
	
	void init(Mnode*, int, int);
	Mnode* append(MFI_tree*, Mnode*, int, int);
};

class Cnode : public FPnode{
public:
	Cnode* par;
	Cnode* leftchild;
	Cnode* rightsibling;
	Cnode* next;
	int level;      //level in MFI-tree
	int count;

public:
	~Cnode();

	void init(Cnode*, int, int, int);
	Cnode* append(CFI_tree*, Cnode*, int, int, int);
};
#endif

