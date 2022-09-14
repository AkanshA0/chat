import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.models import load_model
import csv
from datetime import datetime
import spacy

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

en = spacy.load('en_core_web_sm')
sw_spacy = en.Defaults.stop_words

header = ['ticketId', 'message', 'date', 'userId']
filename = 'new_keywords.csv'
with open(filename, 'w', newline="") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(header)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    print("sentence_words: ")
    print(sentence_words)
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    print("words: ")
    print(words)
    match_count = 0
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
                match_count += 1
    if(match_count == 0):
            bag=[]
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    print("bow: ")
    print(bow)
    if len(bow) == 0:
        return bow
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.6
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        print("class: ")
        print(classes[r[0]])
        print("probability: ")
        print(str(r[1]))
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Bot is running!")

while True:
    message = input("> ")
    sentence_tokens = nltk.word_tokenize(message)
    sentence = ""
    for word in sentence_tokens:
        if word not in sw_spacy:
            sentence = sentence+word+" "
    ints = predict_class(sentence.lower())
    if len(ints) == 0:
        print("invalid input")
        if len(sentence) > 0:
            current_date_time = datetime.now()
            data = ['Q1', message, current_date_time, 'user1']
            with open(filename, 'a', newline="") as file:
                csvwriter = csv.writer(file)
                csvwriter.writerow(data)
            print("written to file")
    else:
        res = get_response(ints,intents)
        print(res)
