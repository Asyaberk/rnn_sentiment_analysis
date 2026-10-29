"""
RNN ile Duygu Dedektifi (Sentiment Analysis)
Problem tanimi:Bir yorumun olumlu mu olumsuz mu olduğunu anlamak(classification)
    IMDB film yorumlari veri seti ile bir metnin duygusal analizi

RNN: tekrarlayan sinir aglari: sirali veriler uzerinde calisiyor, metin gibi verilerle onceki bilgileri hatirlayarak sonraki tahminleri yapmaya calismak
Girdi: film -> cok -> kotuydu
Bellek: 
Çıktı: anlam anlam olumsuz

Veri Seti: IMDB veri seti: film yorumlari(olumlu ve olumsuz)
    - 50000 adet film yorumu
    - 0 negatif 1 pozitif
    - great = 65

plan:


Gerekli kurulumlar:


import libraries:


"""

#import libraries

import numpy as np
import nltk #natural tool kit
import matplotlib.pyplot as plt #gorsellestirme
from nltk.corpus import stopwords #gereksiz kelime listesi
from tensorflow.keras.models import Sequential #base model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.datasets import imdb #veri seti
from tensorflow.keras.preprocessing.sequence import pad_sequences



#stopword (gereksiz kelimeler) listesi belirle
nltk.download("stopwords") #nltk icinden ingilizce stopwords indiriliyor
stop_words = set(stopwords.words("english")) #kucuk ve anlamsiz kelimeler ayrilacak

#model parametreleri
max_features = 10000 #en cok kullanılan 10k kelime

#veri seti yükleyeceğiz load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = max_features) #touple train ve test ayrilmis sekilde veri gelir

#ornek veri incelemesi
original_word_index = imdb.get_word_index()

#sayilardan kelime donusum sozlugu
inv_word_index = {index + 3: word for word, index in original_word_index.items()}
inv_word_index[0] = "<PAD>" #0: bosluk padding
inv_word_index[1] = "<START>" #1: cumle baslangici
inv_word_index[2] = "<UNK>" #2: bilinmeyen kelime

#sayı dizisini kelimelere ceviren fonksiyon
def decode_review(encoded_review):
    return " ".join([inv_word_index.get(i, "?") for i in encoded_review])

movie_index = 0
#ilk egitim verisini yazdir
print("ilk yorum : (sayi dizisi)")
print(X_train[movie_index])

print("ilk yorum: (kelimelerle)")
print(decode_review(X_train[movie_index]))

print(f"Label : {'Pozitif' if y_train[movie_index] == 1 else 'Negatif'}")

#data preprocessing (veri on isleme)



#RNN modeli olusturma



#modelin eğitilmesi training



#evaluation