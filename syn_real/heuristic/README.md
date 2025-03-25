Here are the results for Amazon, Plaintoid, and the OGBL dataset heuristics. We found that:
	1.	For OGBL, since the dataset split is fixed, the variance is zero. Our results are essentially identical to those reported by NCNC.
	2.	For datasets like Cora, there is some discrepancy compared to the reported results in the paper. However, our variance is very low, as we used 10 random seeds.
	3.	For Photo and Computer, there are no suitable results for comparison, but the performance is generally in the range of 30-40%.git st