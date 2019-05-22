print("Starting")
import subprocess
def install(name):
    subprocess.call(['pip', 'install', name])
install('nltk')
install('autocorrect')
from autocorrect import spell
import re
import string
import nltk
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('wordnet')

print('Taking data')
from nltk.corpus import movie_reviews
negative_fileids = movie_reviews.fileids('neg')
positive_fileids = movie_reviews.fileids('pos')
print('Filtering Data')
def neat(word):
    return word.translate(re.sub(r'\d+', '', word).maketrans('', '', string.punctuation)).lower()
stopwords = nltk.corpus.stopwords.words("english")
filtered_words = {neat(word) for word in movie_reviews.words() if not word in stopwords}
filtered_words = {word for word in filtered_words if word != ''}
print('filtered_words:')
print(filtered_words)

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
lemmer = nltk.WordNetLemmatizer()

def build_bag_of_words_features_filtered_normal(words):
    return {neat(word):1 for word in words if neat(word) in filtered_words}
def build_bag_of_words_features_filtered_stem(words):
    return {stemmer.stem(neat(word.lower())):1 for word in words if neat(word) in filtered_words}
def build_bag_of_words_features_filtered_lem(words):
    return {lemmer.lemmatize(neat(word.lower())):1 for word in words if neat(word) in filtered_words}

print('Building Features Normal')
print('pos')
positive_features_normal = [(build_bag_of_words_features_filtered_normal(movie_reviews.words(fileids=[f])), 'pos') for f in positive_fileids]
print('positive_features:',positive_features_normal[0])
print('neg')
negative_features_normal = [(build_bag_of_words_features_filtered_normal(movie_reviews.words(fileids=[f])), 'neg') for f in negative_fileids]
print('negative_features:',negative_features_normal[0])

print('Building Features Stem')
print('pos')
positive_features_stem = [(build_bag_of_words_features_filtered_stem(movie_reviews.words(fileids=[f])), 'pos') for f in positive_fileids]
print('positive_features:',positive_features_stem[0])
print('neg')
negative_features_stem = [(build_bag_of_words_features_filtered_stem(movie_reviews.words(fileids=[f])), 'neg') for f in negative_fileids]
print('negative_features:',negative_features_stem[0])

print('Building Features Lem')
print('pos')
positive_features_lem = [(build_bag_of_words_features_filtered_lem(movie_reviews.words(fileids=[f])), 'pos') for f in positive_fileids]
print('positive_features:',positive_features_lem[0])
print('neg')
negative_features_lem = [(build_bag_of_words_features_filtered_lem(movie_reviews.words(fileids=[f])), 'neg') for f in negative_fileids]
print('negative_features:',negative_features_lem[0])

split = 500

print("NB N")
from nltk.classify import NaiveBayesClassifier
print("Training")
positive_features=positive_features_normal
negative_features=negative_features_normal
NB = NaiveBayesClassifier.train(positive_features[:split]+negative_features[:split])
print("Testing")
print(nltk.classify.util.accuracy(NB, positive_features[:split]+negative_features[:split])*100)
print(nltk.classify.util.accuracy(NB, positive_features[split:]+negative_features[split:])*100)
# NB.show_most_informative_features()

print("NB S")
from nltk.classify import NaiveBayesClassifier
print("Training")
positive_features=positive_features_stem
negative_features=negative_features_stem
NB = NaiveBayesClassifier.train(positive_features[:split]+negative_features[:split])
print("Testing")
print(nltk.classify.util.accuracy(NB, positive_features[:split]+negative_features[:split])*100)
print(nltk.classify.util.accuracy(NB, positive_features[split:]+negative_features[split:])*100)
# NB.show_most_informative_features()

print("NB L")
from nltk.classify import NaiveBayesClassifier
print("Training")
positive_features=positive_features_lem
negative_features=negative_features_lem
NB = NaiveBayesClassifier.train(positive_features[:split]+negative_features[:split])
print("Testing")
print(nltk.classify.util.accuracy(NB, positive_features[:split]+negative_features[:split])*100)
print(nltk.classify.util.accuracy(NB, positive_features[split:]+negative_features[split:])*100)
# NB.show_most_informative_features()

split=500
from sklearn.svm import SVC
from nltk.classify import SklearnClassifier

print('SVC N')
print('Training')
positive_features=positive_features_normal
negative_features=negative_features_normal
train_data = positive_features[:split]+negative_features[:split]
svcc = SklearnClassifier(SVC(), sparse=False).train(train_data)
print('Testing')
print(nltk.classify.util.accuracy(svcc, positive_features[:split]+negative_features[:split])*100)
print(nltk.classify.util.accuracy(svcc, positive_features[split:]+negative_features[split:])*100)

print('SVC S')
print('Training')
positive_features=positive_features_stem
negative_features=negative_features_stem
train_data = positive_features[:split]+negative_features[:split]
svcc = SklearnClassifier(SVC(), sparse=False).train(train_data)
print('Testing')
print(nltk.classify.util.accuracy(svcc, positive_features[:split]+negative_features[:split])*100)
print(nltk.classify.util.accuracy(svcc, positive_features[split:]+negative_features[split:])*100)

print('SVC L')
print('Training')
positive_features=positive_features_lem
negative_features=negative_features_lem
train_data = positive_features[:split]+negative_features[:split]
svcc = SklearnClassifier(SVC(), sparse=False).train(train_data)
print('Testing')
print(nltk.classify.util.accuracy(svcc, positive_features[:split]+negative_features[:split])*100)
print(nltk.classify.util.accuracy(svcc, positive_features[split:]+negative_features[split:])*100)

split = 100

print("DT N")
from nltk.classify import DecisionTreeClassifier
print("Training")
positive_features=positive_features_normal
negative_features=negative_features_normal
DT = DecisionTreeClassifier.train(positive_features[:split]+negative_features[:split])
print("Testing")
print(nltk.classify.util.accuracy(DT, positive_features[:split]+negative_features[:split])*100)
print(nltk.classify.util.accuracy(DT, positive_features[split:]+negative_features[split:])*100)

print("DT S")
from nltk.classify import DecisionTreeClassifier
print("Training")
positive_features=positive_features_stem
negative_features=negative_features_stem
DT = DecisionTreeClassifier.train(positive_features[:split]+negative_features[:split])
print("Testing")
print(nltk.classify.util.accuracy(DT, positive_features[:split]+negative_features[:split])*100)
print(nltk.classify.util.accuracy(DT, positive_features[split:]+negative_features[split:])*100)

print("DT L")
from nltk.classify import DecisionTreeClassifier
print("Training")
positive_features=positive_features_lem
negative_features=negative_features_lem
DT = DecisionTreeClassifier.train(positive_features[:split]+negative_features[:split])
print("Testing")
print(nltk.classify.util.accuracy(DT, positive_features[:split]+negative_features[:split])*100)
print(nltk.classify.util.accuracy(DT, positive_features[split:]+negative_features[split:])*100)

print('ME N')
from nltk.classify import maxent
print("Training")
positive_features=positive_features_normal
negative_features=negative_features_normal
ME = maxent.MaxentClassifier.train(positive_features[:split]+negative_features[:split],max_iter=3)
print("Testing")
print(nltk.classify.util.accuracy(ME, positive_features[:split]+negative_features[:split])*100)
print(nltk.classify.util.accuracy(ME, positive_features[split:]+negative_features[split:])*100)
# ME.show_most_informative_features()

print('ME S')
from nltk.classify import maxent
print("Training")
positive_features=positive_features_stem
negative_features=negative_features_stem
ME = maxent.MaxentClassifier.train(positive_features[:split]+negative_features[:split],max_iter=3)
print("Testing")
print(nltk.classify.util.accuracy(ME, positive_features[:split]+negative_features[:split])*100)
print(nltk.classify.util.accuracy(ME, positive_features[split:]+negative_features[split:])*100)
# ME.show_most_informative_features()

print('ME L')
from nltk.classify import maxent
print("Training")
positive_features=positive_features_lem
negative_features=negative_features_lem
ME = maxent.MaxentClassifier.train(positive_features[:split]+negative_features[:split],max_iter=3)
print("Testing")
print(nltk.classify.util.accuracy(ME, positive_features[:split]+negative_features[:split])*100)
print(nltk.classify.util.accuracy(ME, positive_features[split:]+negative_features[split:])*100)
# ME.show_most_informative_features()

print('Testing for manual input')
tests={'its a good movie storyline best compared to story':'neg'}
test_features=[(build_bag_of_words_features_filtered_lem(each.split()),tests[each]) for each in tests.keys()]
for each in tests.keys():
    print(tests[each],':',each)
print('Accuracy:',nltk.classify.util.accuracy(NB, test_features)*100)
