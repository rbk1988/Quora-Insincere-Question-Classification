# Quora-Insincere-Question-Classification
This repository contains all the necessary details and implementation of Text Classification using Deep Learning.

Let us solve the problem by considering one of the kaggle competitions hosted by Quora.

Problem Statement:-

An existential problem for any major website today is how to handle toxic and divisive content. Quora wants to tackle this problem head-on to keep their platform a place where users can feel safe sharing their knowledge with the world. Quora is a platform that empowers people to learn from each other. On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. A key challenge is to weed out insincere questions -- those founded upon false premises, or that intend to make a statement rather than look for helpful answers. 

For furthur details on problem statement and dataset, refer the link:-https://www.kaggle.com/c/quora-insincere-questions-classification

Objective:-

To build a model that classifies the given question as sincere or insincere using deep learning.

Steps followed:-

1.Understanding and gaining some insights from the given data.

a)Check for any null values

b)Top frequent words that leads to insincere

c)Check for misspelled words

d)Understand the Class distribution

2.Data Wrangling 

a)Convert the text to lower case

b)Contraction mapping

c)Dealing with special characters and punctuations

d)Manual correction of most frequently misspelled words

3.Embeddings 

There are 2 ways to deal with embeddings:-

a)Pretrained embeddings :- GoogleNews-vectors-negative300, glove.840B.300d, paragram_300_sl999, wiki-news-300d-1M 

b)Learning embeddings from the scratch :- word2vec, keras embedding layer

4.Model building:-

a)Neural Networks are capable of learning any non linear function of weights to map the input to output. Thats why, Neural Networks are called as Universal Function Approximators. 

b)The problem with Neural Network is quick generalization and prediction is made by different combinations of inputs. This is a problem 
when the prediction has to made by capturing the sequential information from the input rather than trying out different combinations of the input. 

c)Here comes RNN. But, RNN's suffer from the vanishing gradient due to which it cannot learn the long term dependencies. Thats why we go with LSTMs.

d)Long Short Term Memory models can capture the long range dependencies which is a shortcome of Recurrent Neural Network.

5.Model evaluation:-

a)Stratified K cross fold validation with right metric. 

5.Diagnostic plots

a)Identify whether the model suffers from overfitting, underfitting or good fit with diagnostic plots(No. of epochs vs loss)

6.Hyperparameter Tuning:-

a)

7.Build model with best Hyperparameter set.

Challenges involved:-

1.Data cleaning







