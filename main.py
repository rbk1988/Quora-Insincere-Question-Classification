import pandas as pd
import numpy as np
import seaborn as sns
import re
import nltk
import collections
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers import Bidirectional,GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from matplotlib import pyplot


def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text

def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    return text

def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x


def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def model_build(num_neurons,dropout):
    model=Sequential()
    model.add(Embedding(x_size,300,input_length=max_seq_x,trainable=True)) #Learning embeddings during backpropagation!
    model.add(Bidirectional(LSTM(num_neurons,dropout_rate=dropout,return_sequences=True)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(16,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=["acc"])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=1)
    mc=ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)
    return model


if __name__ == 'main':
    data = pd.read_csv("quora.csv")
    sns.countplot(x="target", data=data)
    data['lowered_question'] = data['text'].apply(lambda x: x.lower())
    contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have" }  

    data['treated_question'] = data['lowered_question'].apply(lambda x: clean_contractions(x, contraction_mapping))

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'",
                     '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '',
                     '³': '3', 'π': 'pi', }
    data['treated_question'] = data['treated_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping)) 

    mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater',
                    'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize',
                    'youtu ': 'youtube ', 'qoura': 'quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do',
                    'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I',
                    'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating',
                    'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota',
                    'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                    'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

    data['treated_question'] = data['treated_question'].apply(lambda x: correct_spelling(x, mispell_dict))

    vocab= build_vocab(data['treated_question'])

    preprocessed_text=[]

    for i in list(data['treated_question']):
      preprocessed_text.append(i.split())
      
    y=list(data['target'])
    words=list(vocab.keys())                            #vocabulary
    x_size=len(words) + 1                               #for padding, add +1
    max_seq_x=max(len(i) for i in preprocessed_text)    #maximum length of a question
    x_word2idx = {w: i + 1 for i, w in enumerate(words)}#Map every word to index
    x_word2idx["PAD"]=0                                 #add pad token to vocabulary
    idx2x_word = {i: w for w, i in x_word2idx.items()}

    x=[]
    for i in preprocessed_text:
      temp=[]
      for j in i:
        temp.append(x_word2idx[j])
      x.append(np.array(temp))
      
    x = pad_sequences(maxlen=max_seq_x, sequences=x, value=x_word2idx['PAD'], padding='post', truncating='post') #pad the sequences to maximum seq and set mask_zero parameter in keras embedding layer to skip these values.
    x_tr,x_val,y_tr,y_val=train_test_split(x,y,test_size=0.3,random_state=2019,stratify=y) #Split data into training and validation data initially

    keras.backend.clear_session()
    model=model_build(64,0.1) #Instantiate the model
    print(model.summary()) #To know about the no. of parameters and layers in model architecture

    #Early stoppage : Traning of neural network is stopped as soon as validation loss keeps increasing.
    #Model checkpoint:- To save the best model after every epoch.
    history = model.fit(np.array(x_tr),np.array(y_tr),batch_size=300,epochs=10,validation_data=(np.array(x_val),np.array(y_val)),verbose=1,callbacks=[es,mc])
    saved_model = load_model('best_model.h5')
    y_pred=model.predict_classes(np.array(x_val))
    print(classification_report(y_val,y_pred))

    #Diagnostic plots!
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    #Tuning dropout regularization and no. of neurons!
    kf = StratifiedKFold(n_splits=3,random_state=5, shuffle=True) #Stratified k cross fold validation
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    neurons = [32,64,128,256]

    #Time consuming!
    for dropout in dropout_rate:
        for num_neurons in neurons:
            validation_score = []
            for train_index, val_index in kf.split(data, target):
                # split data
                x_tr, x_val = x[train_index], x[val_index]
                y_tr, y_val = y[train_index], y[val_index]

                # instantiate model
                model=build_model(num_neurons,dropout_rate)

                history = model.fit(np.array(x_tr),np.array(y_tr),batch_size=300,epochs=10,validation_data=(np.array(x_val),np.array(y_val)),verbose=1,callbacks=[es,mc])

                #validation
                y_pred=model.predict_classes(np.array(x_val))

                fscore=f1_score(y_val, y_pred, average='micro')
            
                # append to appropriate list
                validation_score.append(val_error)

            # generate report
            print('dropout: {:6} | num_neurons:{:6}| mean(val_error): {}'.format(dropout,num_neurons,round(np.mean(validation_score),4)))



