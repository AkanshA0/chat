from flask import Flask, jsonify
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
from flask import Flask, request, jsonify
from flask_socketio import SocketIO,emit
from flask_cors import CORS

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

header = ['ticketId', 'message', 'date', 'userId']
filename = 'new_keywords.csv'
with open(filename, 'w', newline="") as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(header)

en = spacy.load('en_core_web_sm')
sw_spacy = en.Defaults.stop_words

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app,resources={r"/*":{"origins":"*"}})
socketio = SocketIO(app,cors_allowed_origins="*")

@socketio.on("connect")
def connected():
    """event listener when client connects to the server"""
    print(request.sid)
    print("client has connected")
    emit("connect",{"data":f"id: {request.sid} is connected"})

@socketio.on('data')
def handle_message(data):
    """event listener when client types a message"""
    print("data from the front end: ",str(data))
    emit("data",{'data':data,'id':request.sid},broadcast=True)

@socketio.on("disconnect")
def disconnected():
    """event listener when client disconnects to the server"""
    print("user disconnected")
    emit("disconnect",f"user {request.sid} disconnected",broadcast=True)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
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
    if len(bow) == 0:
        return bow
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.6
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

@app.route('/chatbot/<string:userId>/<string:msg>')
def chat(userId,msg):
    sentence_tokens = nltk.word_tokenize(msg)
    sentence = ""
    for word in sentence_tokens:
        if word not in sw_spacy:
            sentence = sentence+word+" "
    ints = predict_class(msg.lower())
    if len(ints) == 0:
        response = {'errorCode':'123','errorMessage':'No action found, please connect to agent'}
        #store message in DB
        if len(msg) > 0:
            current_date_time = datetime.now()
            data = ['Q1', msg, current_date_time, userId]
            with open(filename, 'a', newline="") as file:
                csvwriter = csv.writer(file)
                csvwriter.writerow(data)
    else:
        res = get_response(ints,intents)
        response = {'ticketid': '1234', 'reply': res}
    return jsonify(response)

@app.route('/chatbot/report')
def generateReport():
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        line_count = 0
        key_dict={}
        msg_dict={}
        for row in csv_reader:
            if line_count > 0:
                words = [word for word in row[1].split(" ") if word.lower() not in sw_spacy]
                for w in words:
                   if key_dict.__contains__(w):
                        count = key_dict[w]
                        key_dict[w]=count+1
                        msg = [msg_dict[w],row[1]]
                        msg_dict[w] = msg
                   else:
                       key_dict[w]=1
                       msg_dict[w]=[row[1]]
            line_count += 1
        res = {'keyCount':key_dict, 'keyMessages': msg_dict}
        return json.dumps(res)

if __name__ == '__main__':
    socketio.run(app, debug=True)
