Add links:
* Old README.md got bloated >> Link.
* Initial report >> Link.

# TODO:

The new models, are *NOT* getting better. 
* Add more weights, what can you do with 600k params
* I suggest you add more to the first two layers.

Refactor the code:
	* Make it easier for someone else to join and play with.
	* Remove old files.
	* Write better documentation.

Report writing:
* Finish section on Kronecker Structure.
* Finish section on Van Loan and Pruning.

Coding:
* Make the Prune / Run for 100k
* Seems to be an issue with `.t()`, must use `.contiguous()` 
* Revert the 1250/4000 to H-F  ?DONE
* Run a training for 10 steps only, with very high batch size.  ?DONE

* github issue
* score the 95M
* prune the 95M and see what's up



# DONE:

* Change completely how you decompose and load. [DONE]
	*  Have this structure: fc0 [r, m, n] and fc1 [r, m', n']A
	*  And do the looping inside the forward.
* Check correctness of the decomposition.[DONE]
* Convert old checkpoint 1350 / 4000 to new format. DONE

Runs over the week-end:
* Train the 95M, as much as possible.  [DONE]
	* The do the trick with the embedding   [DONE]

* See how much factors you can pack with 67M.  [DOME]
* Train the max you can with 82M.  [DONE]
