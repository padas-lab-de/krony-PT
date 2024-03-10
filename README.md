Add links:
* Old README.md got bloated >> Link.
* Initial report >> Link.



# TODO:

Report writing:
* Finish section on Kronecker Structure.
* Finish section on Van Loan and Pruning.

Coding:
* Convert old checkpoint 1350 / 4000 to new format. 


# DONE:

* Change completely how you decompose and load. [DONE]
	*  Have this structure: fc0 [r, m, n] and fc1 [r, m', n']A
	*  And do the looping inside the forward.

* Check correctness of the decomposition.[DONE]

Runs over the week-end:
* Train the 95M, as much as possible.  [DONE]
	* The do the trick with the embedding   [DONE]

* See how much factors you can pack with 67M.  [DOME]
* Train the max you can with 82M.  [DONE]
