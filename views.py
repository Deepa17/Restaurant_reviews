from django.shortcuts import render 
#from django.forms import widgets
#from django import url_for,request
#from sklearn.externals import joblib


def home(request):
    return render(request, 'index.html')

def result(request):
  
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) 

    #cleaning the texts
    #Note inernet required to run this cell
    import re #simplify the basic issues not stemming
    import nltk #download the ensemble of stopwords, words that are irrelevant to help predict the results, not included after cleaning the dataset
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer #stemming --> to simplify the  input, say convert loved to love: reducing the dimnesion of the sparse matrix
    corpus = [] #list of cleaned reviews

    for i in range(1000):
        review=re.sub('[^a-zA-Z]',' ', dataset['Review'][i]) #replace all the non alphabets with a space in the column Review of the given row
        review = review.lower().split()#splitting the  review into words and lower case
        #stemming to optmise the dimensionality of the the sparse matrix
        ps = PorterStemmer()#creating an object
        #review = [ps.stem(word) for word in review if word not in set(stopwords.words('english').remove('not'))]
        #the above stmt doens't work directly
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)

        #bag of words
    #tokenisation
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = 1500) #to remove unwanted words that don't help predict the review,
    #the parameters in the paranthesis is to include the 1500 most frequent wrds of english that are found in the list corpus
    #here the size of the sparse matrix(matrices of features) is chosen, i.e the number of requred words
    X = cv.fit_transform(corpus).toarray() #the matrix of features has to be a 2D ARRAY
    y = dataset.iloc[: , -1].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
        
    from sklearn.naive_bayes import MultinomialNB
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)  
    classifier.score(X_test,y_test)  
   
    name = request.POST.get('u_name',"False")
    message = request.POST.get('review', "False")
    data = [message]
    vect = cv.transform(data).toarray()
    my_prediction = classifier.predict(vect)
    return render(request,'result.html',{'prediction':my_prediction, 'uname': name})
    


