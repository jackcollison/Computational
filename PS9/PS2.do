// PS2: Computational Methods
// Author: Stefano Lord Medrano

* Clear workspace
clear all

* Use data set
use "/Users/smlm/Desktop/Desktop - Stefano's MacBook Pro/2nd Year PhD/Econ 899/Problem Sets Solutions/PS9 - Solution/Mortgage_performance_data.dta"

* Create duration variable
generate duration = .
replace duration = 1 if (i_close_0 == 1)
replace duration = 2 if (i_close_0 == 0) & (i_close_1 == 1)
replace duration = 3 if (i_close_0 == 0) & (i_close_1 == 0) & (i_close_2 == 1)
replace duration = 4 if (i_close_0 == 0) & (i_close_1 == 0) & (i_close_2 == 0)
