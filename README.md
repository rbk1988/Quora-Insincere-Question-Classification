# Quora-Insincere-Question-Classification

This repository contains all the necessary details and implementation of classification of text using Deep Learning.

Let us solve the problem by considering one of the kaggle competitions hosted by Quora.

_**Problem Statement**_:-

An existential problem for any major website today is how to handle toxic and divisive content. Quora wants to tackle this problem head-on to keep their platform a place where users can feel safe sharing their knowledge with the world. Quora is a platform that empowers people to learn from each other. On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. A key challenge is to weed out insincere questions -- those founded upon false premises, or that intend to make a statement rather than look for helpful answers. 

For furthur details on problem statement and dataset, refer the link:-https://www.kaggle.com/c/quora-insincere-questions-classification

_**Objective**_:-

To build a model that classifies the given question as sincere or not using deep learning.

_**Steps**_:-

1.**Understanding and gaining some insights from the given data:-**

a) Check for any null values

b) Top frequent words that leads to insincere

c) Check for misspelled words

d) Understand the Class distribution

2.**Data Wrangling:-** 

a) Convert the text to lower case

b) Contraction mapping

c) Dealing with special characters and punctuations

d) Manual correction of most frequently misspelled words

3.**Embeddings** 

There are 2 ways to deal with embeddings:-

a) **Pretrained embeddings** : GoogleNews-vectors-negative300, glove.840B.300d, paragram_300_sl999, wiki-news-300d-1M 

b) **Learning embeddings from the scratch** : word2vec, keras embedding layer

4.**Model building**:-

a) Neural Networks are capable of learning any non linear function of weights to map the input to output. Thats why, Neural Networks are called as _**Universal Function Approximators**_. 

b) The problem with Neural Network is quick generalization and prediction is made by different combinations of inputs. This is a problem 
when the prediction has to made by capturing the sequential information from the input rather than trying out different combinations of the input. 

c) Here comes RNN. But, RNN's suffer from the vanishing gradient due to which it cannot learn the long term dependencies. Thats why we go with LSTMs.

d) __Long Short Term Memory models__ can capture the long range dependencies which is a shortcome of Recurrent Neural Network.

5.**Model evaluation**:-

a) Stratified K cross fold validation with right metric. 

5.**Diagnostic plots**

a) Identify whether the model suffers from overfitting, underfitting or good fit with diagnostic plots(No. of epochs vs loss)

b) In case of overfitting, introduce the regularization i.e dropouts in the neural network which randomly drops out few neurons and due to which the rest of the neurons must step in to handle the representation. This leads to better generalization by making the network less sensitive to specific weights of neurons.

c) Incase of underfitting, bring up the reprensentational capacity of the model.

6.**Hyperparameter Tuning**:-

a) Hyperparameter optimization is the very important to boost up the performance of the model.

b) Tune drop outs, No. of neurons in hidden layers, learning rate using grid search to come up with the best hyperparameter set.

7.**Deploy the model into production** 

_**Challenges Involved**_:-

1.Data cleaning

_**References**_:

Thanks to the below links which helped me with this one.

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

https://www.kaggle.com/sbongo/do-pretrained-embeddings-give-you-the-extra-edge

https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/

https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings






