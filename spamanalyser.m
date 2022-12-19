% Load the dictionary file
dictionary = readtable('dictionary.txt');

% Create a bag-of-words model
bag = bagOfWords(tokenizedText, dictionary);