from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# load the model from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	set=0
	sys=['Unable to launch','Failed to fetch data','Site issue','Code stage','Exception','HRESULT','Automatically set Cleanup','Code stage','Out of memory','Unexpected Exception','Operation failed', 'Internal','Evaluate expression','Code stage','Unexpected Exception','Operation failed','Time out','Failed to Perform']
	bs=['Missing mandatory','fieldsMail Id not found','Invite not found','Template not found','Error in Input File','Not Contain','Field missing','To be processed manually']
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
		if my_prediction==0:
			for x in bs:
				if x in message:
					set=1
					ans=x
					break
		
		if my_prediction==1:
			for x in sys:
				if x in message:
					ans=x
					set=1
					break

	if set==1:
		return render_template('home.html',prediction = my_prediction,sets=set,result=ans)

	if set==0:
		return render_template('home.html',prediction = my_prediction,sets=0,result=0)




if __name__ == '__main__':
	app.run(debug=True)