PHASE 3:
This phase is dedicated to model training and selection. 
run the modelsEvaluation.py will genreate images and tables that help you decide which model is best suited for the task.


ModelsEevaluation.py
the modelsEvaluation.py script has some customizamible parameters. 
If you run it as it is (default parameters are commented) you will reproduce our results. 
Otherwise, you can change:

- the selection of models to be trained
- the resolution of the learning curves
- the cross-validation parameters.
See the file for more details.


You will see that a folder "Latest_experiments" is already filled with trained models. 
These are the models we trained ourselves, and you can generate their learning curves by simply running the script and keeping "Load" = true. 
This will generate some images in Latest_experiments/images.

You can use some utility scripts to process your datasets. They are found in the datasets folder:

PURGE FILE
The purge.py script take in input two files.
The .arff file and the .csv file with the same format as the file used to prepare the datasets.

This purges the .arff from any datapoint contained in the .csv. You can use this to ensure that training and test set are truly separated. The model generator and evaluator will take care of it even if you don't perform this step. 

SEPARATE DATASETS
The separate_datasets.py script allow you to split datasets according to some parameter. We used it to separate the uint8 data from the rest. 