#ifndef _FREQ_ITEMSET_CLASS
#define _FREQ_ITEMSET_CLASS

#include <set>

class FrequentItemset
{
	public:
	
		FrequentItemset(int support);
		
		void insert(int item);
		int support() const;
		std::size_t size() const;
		std::set<int>::iterator begin() const;
		std::set<int>::iterator end() const;
	
	private:
		
		int itemsetSupport;
		std::set<int> itemset;
};

struct FIComparator
{
	bool operator() (FrequentItemset fi1, FrequentItemset fi2) const {return fi1.size() >= fi2.size();}
};

// FISet is a set of frequent itemsets (sorted in decreasing order of size)
typedef std::set<FrequentItemset, FIComparator> FISet;

#endif

