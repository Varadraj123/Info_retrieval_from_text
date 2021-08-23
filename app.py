from flask import Flask,request,render_template
import joblib
from sklearn.neighbors import KDTree
import pandas as pd

app = Flask(__name__)  
  
@app.route('/')  
def customer():  
    return render_template('form.html')  
  
@app.route('/dictionary',methods = ['POST', 'GET'])  
def print_data():  
    person_url={}
    if request.method == 'POST':  
        text_data= request.form.get("text")
        print(f'******************** Input Text is :{text_data}:')
        
        #read the csv file having text of similar data
        person = pd.read_csv(r'C:\Users\shwet\Varad\NLP by Afsaan\datasets\famous_people.csv')

        # loading the model
        vector = joblib.load('tfidf_vector_model.pkl')
        kd_model = joblib.load('kdtree_model.pkl')
            
        tf = vector.transform([text_data]).toarray()
            
        distance, idx = kd_model.query(tf , k = 3)# 'k' is the no of similar results you want to display
            
        for i, value in list(enumerate(idx[0])):
            #print(f"Name: {person['Name'][value]}")
            #print(f"URI: {person['URI'][value]}") 
            person_url[person['Name'][value]]=person['URI'][value]
            
        print(person_url)
              
    return render_template("form.html",final_result = person_url)

if __name__ == '__main__':  
    app.run(debug=True)