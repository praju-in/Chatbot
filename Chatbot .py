#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np


# In[2]:


with open("train_qa","rb") as fp:
    train_data = pickle.load(fp)


# In[3]:


train_data


# In[4]:


with open("test_qa","rb") as fp:
    test_data = pickle.load(fp)


# In[5]:


test_data


# In[6]:


type(train_data)


# In[7]:


type(test_data)


# In[8]:


len(train_data)


# In[9]:


len(test_data)


# In[10]:


train_data[0]


# In[11]:


train_data[0][0]


# In[12]:


' '.join(train_data[0][0])


# In[13]:


' '.join(train_data[0][1])


# In[14]:


train_data[0][2]


# In[15]:


#set up vacbolary
vocab = set()


# In[16]:


all_data = test_data + train_data


# In[17]:


type(all_data)


# In[18]:


all_data


# In[19]:


for a in all_data:
    print(a)
    break


# In[20]:


for story, question, answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))


# In[21]:


vocab.add('yes')


# In[22]:


vocab.add('no')


# In[23]:


vocab


# In[24]:


len(vocab)


# In[25]:


vocab_len = len(vocab)+1


# In[26]:


max_story_len = max([len(data[0]) for data in all_data])
max_story_len


# In[27]:


max_question_len = max([len(data[1]) for data in all_data])
max_question_len


# In[ ]:


pip install keras


# In[ ]:


pip install tensorflow


# In[28]:


#vectorize
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


# In[29]:


tokenizer = Tokenizer(filters = [])


# In[30]:


tokenizer.fit_on_texts(vocab)


# In[31]:


tokenizer.word_index


# In[32]:


train_story_text = []
train_question_text = []
train_answers = []

for story, question, answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)


# In[33]:


train_story_seq = tokenizer.texts_to_sequences(train_story_text)


# In[34]:


len(train_story_text)


# In[35]:


len(train_story_seq)


# In[36]:


train_story_seq


# In[37]:


train_story_text


# In[38]:


def vectorize_stories(data, word_index = tokenizer.word_index,
                     max_story_len = max_story_len, max_ques_len = max_question_len):
    X = []
    Xq = []
    Y = []
    
    for story, query, answer in data:
        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in query]
        y = np.zeros(len(word_index)+1)
        y[word_index[answer]]=1
        
        X.append(x)
        Xq.append(xq)
        Y.append(y)
        
        return(pad_sequences (X , maxlen = max_story_len),
              pad_sequences (Xq , maxlen = max_ques_len),
              np.array(Y))


# In[39]:


inputs_train, queries_train, answers_train = vectorize_stories(train_data)
inputs_test, queries_test, answers_test = vectorize_stories(test_data)


# In[40]:


inputs_train


# In[41]:


queries_test


# In[42]:


inputs_test


# In[43]:


queries_test


# In[44]:


answers_test 


# In[45]:


tokenizer.word_index['yes']


# In[46]:


tokenizer.word_index['no']


# In[47]:


from keras.models import Sequential,Model
from keras.layers import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM


# In[48]:


input_sequence = Input((max_story_len,))
quention = Input((max_question_len,))


# In[49]:


#input encoder m
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim = vocab_len, output_dim = 64))
input_encoder_m.add(Dropout(0.3))


# In[50]:


#input encoder c
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim = vocab_len, output_dim = max_question_len))
input_encoder_c.add(Dropout(0.3))


# In[51]:


#quetion encoder 
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim = vocab_len, output_dim= 64, input_length = max_question_len))
question_encoder.add(Dropout(0.3))


# In[52]:


#encode the sequences
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(quention)


# In[53]:


match = dot([input_encoded_m,question_encoded], axes = (2,2))
match = Activation('softmax')(match)


# In[54]:


response = add([match,input_encoded_c])
response = Permute((2,1))(response)


# In[55]:


#Concatenate
answer = concatenate([response,question_encoded])


# In[56]:


answer


# In[57]:


answer = LSTM(32)(answer)


# In[58]:


answer = Dropout(0.5)(answer)
answer = Dense(vocab_len)(answer)


# In[59]:


answer = Activation('softmax')(answer)


# In[60]:


model = Model([input_sequence, quention], answer)
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',metrics = ['accuracy'])


# In[61]:


model.summary()


# In[62]:


history = model.fit([inputs_train,queries_train],answers_train,
                   batch_size = 30, epochs = 37,
                   validation_data = ([inputs_test,queries_test],answers_test))


# In[63]:


import matplotlib.pyplot as plt

print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("epochs")
plt.ylabel("Accuracy")


# In[64]:


#ssave
model.save("chatbot_model")


# In[65]:


#Evalution on the test set
model.load_weights("chatbot_model")


# In[66]:


pred_results = model.predict(([inputs_test, queries_test]))


# In[67]:


test_data[0][0]


# In[68]:


story = ' '.join(word for word in test_data[25][0])


# In[69]:


story


# In[70]:


query = ' '.join(word for word in test_data[25][1])


# In[71]:


query


# In[72]:


test_data[25][2]


# In[73]:


val_max = np.argmax(pred_results[0])

for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key
        
print("Predicted Answer is", k)
print("Probability of certainity", pred_results[0][val_max])


# In[74]:


vocab


# In[75]:


story = "Daniel dropped the milk . john discarded football in office . john went to bedroom . "
story.split()


# In[76]:


my_question = "is john in bedroom ?"


# In[77]:


my_question.split()


# In[78]:


my_data = [(story.split(), my_question.split(), 'yes')]


# In[79]:


my_story, my_ques, my_ans = vectorize_stories(my_data)


# In[80]:


pred_results = model.predict(([my_story, my_ques]))


# In[81]:


val_max = np.argmax(pred_results[0])

for key,val in tokenizer.word_index.items():
    if val == val_max:
        k = key
        
print("Predicted Answer is", k)
print("Probability of certainity",pred_results[0][val_max])


# In[82]:


type(inputs_train)


# In[ ]:




