Both models are contained and run within the main.m MATLAB file. The spamanalyser.m file is not used.

The Naive Bayes model is worked manually (could not get bagOfWords working with MATLAB text analysis module).
The TreeBagger function is used to create the RF model. See: https://uk.mathworks.com/help/stats/treebagger.html

The data has been preprocessed using Python (3.10) and the following libraries in the preproccessing.ipynb file:
csv, re, pandas, string, nltk, nltk.corpus stopwords, wordcloud WordCloud, matplotlib.pyplot, seaborn.

MATLAB r2022b version is used. 

All files should be run within the same folder.

The train_features2500.txt and test_features2500.txt should be used, as well as the train_labels.txt and test_labels.txt.

The "//spam_train.csv and //spam_test.csv" were all created during testing and ultimately not used within MATLAB.

Created by:
Sam Moradi-Balf
As part of MSc Data Science, Machine Learning Module
City, University of London
