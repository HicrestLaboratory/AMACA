



<br/><br/><br/>
The purge.py script take in input two files.
<br/>
The .arff file and the .csv file with the same format as the file used to prepare the datasets.
<br/>


This purges the .arff from any datapoint contained in the .csv. You can use this to ensure that training and test set are truly separated. The model generator and evaluator will take care of it even if you don't perform this step. 
```
