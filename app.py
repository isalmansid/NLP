from pyresparser import ResumeParser
from docx import Document
from flask import Flask, render_template, redirect, request
import numpy as np
import pandas as pd
import re
from ftfy import fix_text
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import spacy
#nlp = spacy.load("en_core_web_sm")
#nlp = en_core_web_sm.load()

spacy.load("en_core_web_sm")
#spacy.load('en_core_web_sm')


#import en_core_web_sm


stopw = set(stopwords.words('english'))

#stopw = nltk.corpus.stopwords.words('english')
#print(stopwords[:10])

#nl = nltk.download('stopwords')

#df = pd.read_csv('mum.csv',sep=";", encoding='cp1252',error_bad_lines=False)







#
#

#
#df = pd.read_csv('mum.csv')

#
#
#


df = joblib.load('mum.pkl')
#df = pd.DataFrame()



#df = pd.read_csv('mum.csv', sep='|', encoding='cp1252')

df['test'] = df['Job_Description'].apply(
    lambda x: ' '.join([word for word in str(x).split() if len(word) > 2 and word not in (stopw)]))

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('model.html')


@app.route("/home")
def home():
    return redirect('/')


@app.route('/submit', methods=['POST'])
def submit_data():
    if request.method == 'POST':

        f = request.files['userfile']
        f.save(f.filename)
        try:
            doc = Document()
            with open(f.filename, 'r') as file:
                doc.add_paragraph(file.read())
                doc.save("text.docx")
                data = ResumeParser('text.docx').get_extracted_data()

        except:
            data = ResumeParser(f.filename).get_extracted_data()
        resume = data['skills']
        print(type(resume))

        skills = []
        skills.append(' '.join(word for word in resume))
        org_name_clean = skills

        def ngrams(string, n=3):
            string = fix_text(string)  # fix text
            string = string.encode("ascii", errors="ignore").decode()  # remove non ascii chars
            string = string.lower()
            chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
            rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
            string = re.sub(rx, '', string)
            string = string.replace('&', 'and')
            string = string.replace(',', ' ')
            string = string.replace('-', ' ')
            string = string.title()  # normalise case - capital at start of each word
            string = re.sub(' +', ' ', string).strip()  # get rid of multiple spaces and replace with a single
            string = ' ' + string + ' '  # pad names for ngrams...
            string = re.sub(r'[,-./]|\sBD', r'', string)
            ngrams = zip(*[string[i:] for i in range(n)])
            return [''.join(ngram) for ngram in ngrams]

        vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
        tfidf = vectorizer.fit_transform(org_name_clean)
        print('Vecorizing completed...')

        def getNearestN(query):
            queryTFIDF_ = vectorizer.transform(query)
            distances, indices = nbrs.kneighbors(queryTFIDF_)
            return distances, indices

        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
        unique_org = (df['test'].values)
        distances, indices = getNearestN(unique_org)
        unique_org = list(unique_org)
        matches = []
        for i, j in enumerate(indices):
            dist = round(distances[i][0], 2)




            temp = [dist]
            matches.append(temp)
        matches = pd.DataFrame(matches, columns=['Match confidence'])
        df['match'] = matches['Match confidence']
        df1 = df.sort_values('match')
        df2 = df1[['Position', 'Company', 'Location','url',]].head(20).reset_index()











        # Function to convert file path into clickable form.
        import os as o
        def fun(path):

            # returns the final component of a url
            f_url = o.path.basename(path)

            # convert the url into link
            return '<a href="{}">{}</a>'.format(path, f_url)

        # applying function "fun" on column "location".
        df2 = df2.style.format({'url': fun})

    # return  'nothing'
    return render_template('model.html', tables=[df2.to_html(classes='job')], titles=['na', 'Job'])


# if __name__ =="__main__":




# app.run()

#from waitress import serve

#serve(app, host="0.0.0.0", port=8080)
#serve(app)

#app.listen(process.env.PORT | 3000, function())


#if __name__ == "__main__":
 #   app.run(debug=True)

#from waitress import serve
#serve(app, host="0.0.0.0", port=8080)

#server = http.createServer(app)

if __name__ == "__main__":
    app.debug = True
    app.run()
    #FLASK_APP = app
    #FLASK_ENV = "development"
    # app.run(debug=True)
    #app.run('localhost', 5000, debug=True)

    #app.run(host='0.0.0.0', debug=True)


    #FLASK_APP = app

    #FLASK_ENV = "development"




    #app.run('localhost', 5000, debug=True)
