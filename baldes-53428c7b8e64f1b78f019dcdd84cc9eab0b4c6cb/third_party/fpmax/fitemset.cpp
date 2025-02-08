#include "fitemset.h"

FrequentItemset::FrequentItemset(int support) : itemsetSupport(support), itemset()
{
}

void FrequentItemset::insert(int item)
{
	itemset.insert(item);
}

int FrequentItemset::support() const
{
	return itemsetSupport;
}

std::size_t FrequentItemset::size() const
{
	return itemset.size();
}

std::set<int>::iterator FrequentItemset::begin() const
{
	return itemset.begin();
}

std::set<int>::iterator FrequentItemset::end() const
{
	return itemset.end();
}
