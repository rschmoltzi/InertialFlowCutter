## Usage of test scripts

The test scripts test the cuts produced on the graphs placed in the order specified in config.py. The default path is ../affinity/walshaw/. To run the test scripts execute `python3 -B test.py ordering_alg [amount_orderings]`. ordering_alg is an element of {affinity, plm, alg_dist, fa2, accumulated, asc_affinity, asc_plm, asc_accumulated} and describes the ordering algorithm used. Since the code relies on dictionaries returning the keys in insertion order, at least python 3.7 is needed.
