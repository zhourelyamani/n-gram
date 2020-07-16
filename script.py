#1-Split the dataset into a training and a testing subset. Use the category “title” for the testing set and the categories “comment” and “post” for the training set.
import pandas as pd
import numpy as np
df = pd.read_csv('D:/asa/mywork/generatedstackexchange_812k.csv')
df['category'].unique()
title = df['category'] == 'title'
posts = df['category'] == 'post'
commt = df['category'] == 'comment'
train_df = df[posts | commt]
test_df=df[title]
test_df.iloc[0]
train_df.iloc[91752]
#2-Build the matrix of prefix—word frequencies.

    #Use the ngrams function from nltk.utils to generate all n-grams from the corpus
    #Set the following left_pad_symbol = <s> and right_pad_symbol = </s>
from nltk.util import ngrams
from collections import Counter, defaultdict
import numpy as np
import math
prefix_word_counts = defaultdict(Counter)
prefix_word_freq = dict()
vocab = set()
rng = np.random.default_rng()
def bigram_counts(s_string):
    global prefix_word_counts
    s_list = s_string.split(' ')
    vocab.update({*s_list})
    
    for tg in ngrams(s_list, 3, pad_left=True, pad_right=True, right_pad_symbol='</s>', left_pad_symbol='<s>'):
        prefix = tg[:-1]
        word = tg[-1]
        if not prefix in prefix_word_counts:
            prefix_word_counts[prefix] = Counter()
        prefix_word_counts[prefix][word] += 1
    print(prefix_word_counts)
    
def counts_to_freq():
    for prefix, words in prefix_word_counts.items():
        tot = sum(words.values())
        prefix_word_freq[prefix]= { i:words[i]/tot for i in words }
 
#3- Write a text generation function:

    takes a bigram as input and generates the next token
    iteratively slide the prefix over the generated text so that the new prefix includes the most recent token; generates the next token
    to generate each next token, sample the list of words associated with the prefix using the probability distribution of the prefix
    stop the text generation when a certain number of words have been generated or the latest token is a </s>.
 
        
def next_word(prefix):
    nwd = prefix_word_freq[tuple(prefix)]
    nw  = rng.choice(list(nwd.keys()), 1, p=list(nwd.values()))[0]
    return nw

def generate_text(start, wc=0):
    
    if type(start) == str:
        start = start.split(' ')
    prefix = start[-2:]
    
    text = prefix 
    
    one_sentence = False
    if wc==0:
        one_sentence = True
        wc = 50
    
    while wc > 0:
        nw = next_word(prefix)
        text.append(nw)
        wc -=1
        if one_sentence and text[-1] == '</s>':
            break
        prefix = text[-2:]
    return text
  
#4- Write a function that can estimate the probability of a sentence and use it to select the most probable sentence out of several candidate sentences.

    #Split the sentence into trigrams and use the chain rule to calculate the probability of the sentence as a product of the bigrams—tokens probabilities
  
 def get_probs(s):
    if type(s) == str:
        s = s.split(' ') 
    
    default_prob  = 0.005 # default value when the prefix doses not exist in corpus
    probs  = []
    tgrams = ngrams(s, 3,  pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')
    for tg in tgrams:
        prefix = tg[:-1]
        word = tg[-1]
        if not prefix in prefix_word_freq:
            probs.append(default_prob)
        else:
            pb = prefix_word_freq[prefix][word] if word in prefix_word_freq[prefix] else default_prob
            probs.append(pb)
    return probs
    
  def prob_of_sent(s):
    probs = get_probs(s)
    
    lprobs = [math.log(p) for p in probs]
    return math.exp(sum(lprobs))
 #5-Implement the perplexity scoring function for a given sentence and for the training corpus.
  def perplexity(s):     
    lp = [math.log(p) for p in get_probs(s)]
    N  = len(lp)
    return math.exp((-1/N) * sum(lp))
    
#6- Implement Additive Laplace smoothing to give a non-zero probability to missing prefix—token combinations when calculating perplexity

def laplace_smoothed_prob(s):
    global vocab
    probs = []
    vocab_size = len(vocab)
    delta  = 0.25  

    if type(s) == str:
        s = s.split(' ')  
    
    tgrams = ngrams(s, 3,  pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')

    for tg in tgrams:
        prefix = tg[:-1]
        word = tg[-1] 
        prefix_word_count = delta
        prefix_count = delta * vocab_size
        if prefix in prefix_word_counts:
            prefix_count += sum(prefix_word_counts[prefix].values())
            prefix_word_count += prefix_word_counts[prefix][word]
        probs.append(prefix_word_count/prefix_count)
    return probs
    
    
def lps_prob_of_sent(s):
    probs = laplace_smoothed_prob(s)
    
    lprobs = [math.log(p) for p in probs]
    return math.exp(sum(lprobs))
