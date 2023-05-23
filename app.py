from detector import predict_news_rating,predict_label
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/getinput', methods=['GET','POST'])
def get_input():
    text = request.form['textBar']
    input =  predict_news_rating(text) 
    return render_template('result.html', input=input)

if __name__ == '__main__':
    app.run(debug=True)
