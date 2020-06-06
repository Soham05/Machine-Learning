
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import difflib


# In[2]:


def soundex(name):

    """
    The Soundex algorithm assigns a 1-letter + 3-digit code to strings,
    the intention being that strings pronounced the same but spelled
    differently have identical encodings; words pronounced similarly
    should have similar encodings.
    """

    soundexcoding = [' ', ' ', ' ', ' ']
    soundexcodingindex = 1

    #           ABCDEFGHIJKLMNOPQRSTUVWXYZ
    mappings = "01230120022455012623010202"

    soundexcoding[0] = name[0].upper()

    for i in range(1, len(name)):
        c = ord(name[i].upper()) - 65
        if c >= 0 and c <= 25:
            if mappings[c] != '0':
                if mappings[c] != soundexcoding[soundexcodingindex-1]:
                    soundexcoding[soundexcodingindex] = mappings[c]
                    soundexcodingindex += 1

                if soundexcodingindex > 3:
                    break

    if soundexcodingindex <= 3:
        while(soundexcodingindex <= 3):
            soundexcoding[soundexcodingindex] = '0'
            soundexcodingindex += 1

    return ''.join(soundexcoding)


# In[3]:


with open('Boys.txt', 'r') as f:
    boy_names = f.read().splitlines()
with open('Girls.txt', 'r') as f:
    girl_names = f.read().splitlines()


# In[4]:


boys_score = []
for name in boy_names:
    boys_score.append(soundex(name))
girls_score = []
for name in girl_names:
    girls_score.append(soundex(name))   


# In[5]:


boy_df = pd.DataFrame({'names':boy_names,'score':boys_score})
girl_df = pd.DataFrame({'names':girl_names,'score':girls_score})


# In[6]:


def find_similar_name(name,gender):
    if gender in ['m','M','male','MALE','Male']:
        temp_score = soundex(name)
        if temp_score in list(boy_df['score']):
            return(list(boy_df['names'].loc[boy_df['score'] == temp_score]))
        elif difflib.get_close_matches(name,list(boy_df['names'])):
            return(difflib.get_close_matches(name,list(boy_df['names'])))
        else:
            return(name[:4])
        
    if gender in ['f','F','female','FEMALE','Female']:
        temp_score = soundex(name)
        if temp_score in list(girl_df['score']):
            return(list(girl_df['names'].loc[girl_df['score'] == temp_score]))
        else:
            return(difflib.get_close_matches(name,list(girl_df['names'])))


# In[7]:


if __name__ == '__main__':
    gender = input('Male/Female: ')
    name = input('Enter Name: ')
    print(find_similar_name(name,gender))

