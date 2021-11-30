import numpy as np
import pandas as pd
import nltk
import random
from sklearn.model_selection import train_test_split


class SimpleAttack():
  def __init__(self):

    enron_emails = pd.read_csv('data/enron_cleaned.csv')
    enron_emails = enron_emails.fillna('')
    enron_ham = enron_emails[enron_emails['target'] == 0]
    enron_spam = enron_emails[enron_emails['target'] == 1]
    
    enron_ham = enron_ham['clean_msg_no_lst'].tolist()
    enron_ham = str(enron_ham).replace("'",'')
    enron_ham = str(enron_ham).replace(",",'')
    
    tokens = nltk.word_tokenize(enron_ham)
    bigram_fd_ham = nltk.FreqDist(nltk.bigrams(tokens))

    enron_spam = enron_spam['clean_msg_no_lst'].tolist()
    enron_spam = str(enron_spam).replace("'",'')
    enron_spam = str(enron_spam).replace(",",'')

    tokens = nltk.word_tokenize(enron_spam)
    bigram_fd_spam = nltk.FreqDist(nltk.bigrams(tokens))

    self.enron_emails = enron_emails[['clean_msg_no_lst','target']]
    self.most_common_ham = list(bigram_fd_ham.most_common(100))
    self.most_common_spam = list(bigram_fd_spam.most_common(100))

    
  def dictionary_attack_pure_ham(self, malicious_content, length=50, number_of_spam=100):
    spam_list = []
    targets = np.ones((length,),dtype=int).tolist()
    length = int(random.choice(np.linspace(5,length*2)))
    for num in range(number_of_spam):
      spam = malicious_content
      for iter in range(length):
        bigram = random.choice(self.most_common_ham)
        spam += '{} '.format(bigram[0][0])
        spam += '{} '.format(bigram[0][1])
      spam_list.append(spam.strip())

    attack_df = pd.DataFrame(zip(spam_list, targets), columns=['clean_msg_no_lst', 'target'])
    return attack_df

  def inject_dictionary_words(self, email, email_tol, inject_tol):
    if email['target'] == 0:
      return email
    if random.randint(0,100) > email_tol:
      return email
    
    text = str(email['clean_msg_no_lst']).split(' ')
    num_injections = int(len(text) * inject_tol)
    bigrams = random.choices(self.most_common_ham, k=num_injections)

    for bigram in bigrams:
      ham = '{} {}'.format(bigram[0][0], bigram[0][1])
      text_len = len(text)
      idx = random.choice(np.arange(text_len).tolist())
      text.insert(idx, ham)

    email['clean_msg_no_lst'] = ' '.join(map(str, text))
    return email

  def dictionary_attack_from_spam(self, percent_emails=15, percent_dict_in_email=20):
    _, X_test, _, _ = train_test_split(self.enron_emails[['clean_msg_no_lst', 'target']], self.enron_emails.target, random_state = 42, test_size = 0.2)
    X_test = X_test.apply(lambda email: self.inject_dictionary_words(email, percent_emails, percent_dict_in_email), axis=1)
    return X_test


  def tokenize(self, email, tokenize_tol, spam_words):
    if email['target'] == 0:
      return email
    words = str(email['clean_msg_no_lst']).split(' ')
    for idx in range(len(words)):
      if random.randint(0,100) > tokenize_tol: continue
      if words[idx] in spam_words:
        spam_word = list(words[idx])
        spam_word = ' '.join(map(str, spam_word))
        words[idx] = spam_word
    
    email['clean_msg_no_lst'] = ' '.join(map(str, words))
    return email

  def tokenize_spam(self, percent_tokenize=20):
    spam_words = []
    for bigram in self.most_common_spam:
      spam_words.append(bigram[0][0])
      spam_words.append(bigram[0][1])
    
    _, X_test, _, _ = train_test_split(self.enron_emails[['clean_msg_no_lst', 'target']], self.enron_emails.target, random_state = 42, test_size = 0.2)
    X_test = X_test.apply(lambda email: self.tokenize(email, percent_tokenize, spam_words), axis=1)
    return X_test

attack = SimpleAttack()
X_test = attack.tokenize_spam()

# print(X_test.head())






