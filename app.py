#Imports
from flask import Flask, render_template, request, jsonify
import nltk
import datetime
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

stemmer = LancasterStemmer()
seat_count = 50

with open("intents.json", encoding='utf-8-sig') as file:
	data = json.load(file)
with open("data.pickle","rb") as f:
	words, labels, training, output = pickle.load(f)

#Function to process input
def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]
	
	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i,w in enumerate(words):
			if w == se:
				bag[i] = 1

	return np.array(bag)

tf.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]), activation = "softmax")
net = tflearn.regression(net)

#Loading existing model from disk
model = tflearn.DNN(net)
model.load("model.tflearn")


app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')


@app.route('/get')
def get_bot_response():
    global seat_count
    message = request.args.get('msg')
    if message:
        message = message.lower()
        results = model.predict([bag_of_words(message,words)])[0]
        result_index = np.argmax(results)
        tag = labels[result_index]
        print(results)
        if results[result_index] > 0.5:
            for tg in data['intents']:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    break
            response = random.choice(responses)
        else:
            response = "Glad you asked! What do you need my help with?", "I’m listening. Could you repeat your message with a bit more detail so that I can get you the right answer?"
        return str(response)
    return "Missing Data!"

	
if __name__ == "__main__":
	app.run()