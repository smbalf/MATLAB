import fitcnb.*

%%
number_of_training_messages = 1046;
number_of_tokens = 2500;

%%
% Load train_features text file containing top 2500 freq words as per
% the freq_word_list
M = dlmread('train_features2500.txt', ' ');
% create sparse matrix
spmatrix = sparse(M(:, 1), M(:, 2), M(:, 3), number_of_training_messages, number_of_tokens);
train_matrix = full(spmatrix); % the bag of words for the training data

%%
% load train_labels
train_labels = dlmread('train_labels.txt', ' '); % 1046 (1=ham, 2=spam)

%%
% spam ham message indices
spam_indices = find(train_labels == 2); %523 
ham_indices = find(train_labels == 1); %523

%%
% word count for each sms (i.e 13 for row 1, 6 for row 2) 
message_lengths = sum(train_matrix, 2);

%%
% split wordcounts of spam and ham messages
spam_wc = sum(message_lengths(spam_indices)); %9039 words
ham_wc = sum(message_lengths(ham_indices)); %4241 words
%spam messages total more words, more than double nonspam word counts

%%
% probability of respective word in spam/ham
prob_token_spam = ( sum(train_matrix(spam_indices, :)) + 1) ./ (spam_wc + number_of_tokens);
prob_token_ham = ( sum(train_matrix(ham_indices, :)) + 1) ./ (ham_wc + number_of_tokens);
% the prob should progressively get smaller as the freq_word_list was
% created in a descending order, may not be 1:1 but should generally follow
% this order...

%%
% importing the test features!
N = dlmread('test_features2500.txt', ' ');
% creating a sparse matrix for the test features
test_matrix = sparse(N(:, 1), N(:, 2), N(:, 3), 444, 2500); %444 rows 2500 columns

%%
number_of_test_messages = size(test_matrix, 1); % 444 test messages 
number_of_test_tokens = size(test_matrix, 2); % 2496 test tokens trying manual override
%%
%empty array to be filled of length number of test messages (444)
output = zeros(number_of_test_messages, 1);

%%
% prob. that text is spam, total spam count divided by total training set
prob_spam = length(spam_indices)/number_of_training_messages;
% 0.5 as ham spam count is equal in the training data set

%%
test_spam = log(prob_token_spam')
test_ham = log(prob_token_ham')

%prob_token ham has 2500 columns vs test_matrix having 2496...
%cant multiply the two... need fix. (FIXED)

%%
% calc prob for ham n spam texts for all texts in test
% prob_token_spam/ham' transposes row vector to column vector
prob_a_loga = test_matrix*log(prob_token_spam') + log(prob_spam);
prob_b_loga = test_matrix*log(prob_token_ham') + log(1-prob_spam);

%%
output = prob_a_loga > prob_b_loga;

%%
% load the test_labels text file to compare model to test set values 
test_labels = dlmread('test_labels.txt', ' ');

%%
% identify incorrect classified
incorrect_classification =  sum(xor(output, test_labels));

%%
%print error/accuracy percentage.... 
error = incorrect_classification/number_of_test_messages
accuracy = 1 - error

%0.491
%0.509

%%
% probability of respective word in spam/ham
% Laplace smoothing parameter of 0.5
prob_token_spam = ( sum(train_matrix(spam_indices, :)) + 0.5) ./ (spam_wc + number_of_tokens);
prob_token_ham = ( sum(train_matrix(ham_indices, :)) + 0.5) ./ (ham_wc + number_of_tokens);

%%
% calc prob for ham n spam texts for all texts in test
% prob_token_spam/ham' transposes row vector to column vector
prob_a_loga = test_matrix*log(prob_token_spam') + log(prob_spam);
prob_b_loga = test_matrix*log(prob_token_ham') + log(1-prob_spam);

%%
output = prob_a_loga > prob_b_loga;

%%
% load the test_labels text file to compare model to test set values 
test_labels = dlmread('test_labels.txt', ' ');

%%
% identify incorrect classified
incorrect_classification =  sum(xor(output, test_labels));

%%
%print error/accuracy percentage.... 
error = incorrect_classification/number_of_test_messages
accuracy = 1 - error

%0.486
%0.514

%%
% probability of respective word in spam/ham
% Laplace smoothing parameter of 0.5
prob_token_spam = ( sum(train_matrix(spam_indices, :)) + 1.5) ./ (spam_wc + number_of_tokens);
prob_token_ham = ( sum(train_matrix(ham_indices, :)) + 1.5) ./ (ham_wc + number_of_tokens);

%%
% calc prob for ham n spam texts for all texts in test
% prob_token_spam/ham' transposes row vector to column vector
prob_a_loga = test_matrix*log(prob_token_spam') + log(prob_spam);
prob_b_loga = test_matrix*log(prob_token_ham') + log(1-prob_spam);

%%
output = prob_a_loga > prob_b_loga;

%%
% load the test_labels text file to compare model to test set values 
test_labels = dlmread('test_labels.txt', ' ');

%%
% identify incorrect classified
incorrect_classification =  sum(xor(output, test_labels));

%%
%print error/accuracy percentage.... 
error = incorrect_classification/number_of_test_messages
accuracy = 1 - error

%0.4955
%0.5045

%%
% probability of respective word in spam/ham
% Laplace smoothing parameter of 0.5
prob_token_spam = ( sum(train_matrix(spam_indices, :)) + 5) ./ (spam_wc + number_of_tokens);
prob_token_ham = ( sum(train_matrix(ham_indices, :)) + 5) ./ (ham_wc + number_of_tokens);

%%
% calc prob for ham n spam texts for all texts in test
% prob_token_spam/ham' transposes row vector to column vector
prob_a_loga = test_matrix*log(prob_token_spam') + log(prob_spam);
prob_b_loga = test_matrix*log(prob_token_ham') + log(1-prob_spam);

%%
output = prob_a_loga > prob_b_loga;

%%
% load the test_labels text file to compare model to test set values 
test_labels = dlmread('test_labels.txt', ' ');

%%
% identify incorrect classified
incorrect_classification =  sum(xor(output, test_labels));

%%
%print error/accuracy percentage.... 
error = incorrect_classification/number_of_test_messages
accuracy = 1 - error

%0.5114
%0.4887

%%
%
% RANDOM FOREST
%

%%
% Convert training labels to categorical array
RFtrain_labels = categorical(train_labels);

%%
% Create a random forest classifier with 100 decision trees
B = TreeBagger(100,train_matrix, RFtrain_labels);

%%
test_matrix_full = full(test_matrix);

%%
% Make predictions on test data
[predictions,scores] = B.predict(test_matrix_full);
%%
pred_cat = categorical(predictions);

%%
% Convert test labels to categorical array
test_labels_cat = categorical(test_labels);

%%
% Calculate the confusion matrix
confusion_matrix = confusionmat(test_labels_cat, pred_cat);

%%
% Calculate the overall accuracy
accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix(:))

%%
% Define the labels for TP, FP, FN, TN
labels = {'True Positives', 'False Positives', 'False Negatives', 'True Negatives'};

% Extract the values for TP, FP, FN, TN from the confusion matrix
values = [confusion_matrix(1,1), confusion_matrix(1,2), confusion_matrix(2,1), confusion_matrix(2,2)];

% Create the bar plot
figure;
bar(values);

% Set the labels for the x-axis
set(gca, 'XTickLabel', labels);

% Set the axis labels
xlabel('Prediction');
ylabel('Count');

% Add a title
title('Confusion Matrix');

