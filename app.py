import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']

# Following is lines can be used to get accuracy score with accuracy_score
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #allocates 20% of data to testing
# predictions = model.predict(X_test)
# score = accuracy_score(y_test, predictions)

model = DecisionTreeClassifier()
model.fit(X, y)

# joblib.dump(model, 'music-recommender.joblib') # use this to save model with given name
# model = joblib.load(model, 'music-recommender.joblib') -> once model is saved, load it to use to make predictions
# predictions = model.predict([[21, 1]]) -> predictions contains predicted genre of a 21 year old male

tree.export_graphviz(model ,out_file='music-recommender.dot',
                    feature_names=['age', 'gender'],
                    class_names=sorted(y.unique()),
                    label='all',
                    rounded=True,
                    filled=True)