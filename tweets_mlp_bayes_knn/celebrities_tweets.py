import itertools
import string
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB


# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('omw-1.4')

lemmatiser = WordNetLemmatizer()

celebrities_tweets = pd.read_csv("celebrities_tweets.csv")
stopwords = stopwords.words("english")
mystopwords = ['t', 'https', 'http', 'co', 'm', 'n', 's', 'rt', 'amp']
stopwords.extend(mystopwords)


def text_process(tex):
    # 1. Removal of Punctuation Marks
    nopunct=[char for char in tex if char not in string.punctuation]
    nopunct=''.join(nopunct)
    # 2. Lemmatisation
    a = ''
    for i in range(len(nopunct.split())):
        b = lemmatiser.lemmatize(nopunct.split()[i], pos="v")
        a = a+b+' '
    # 3. Removal of Stopwords
    return " ".join([word for word in a.split() if word.lower() not in stopwords])


for i in range(len(celebrities_tweets)):
    celebrities_tweets.at[i, 'text'] = text_process(celebrities_tweets.at[i, 'text'])

y = celebrities_tweets['name']
labelencoder = LabelEncoder()

celebrities_tweets["name_n"] = labelencoder.fit_transform(y)
classes = celebrities_tweets["name_n"]
inputs = celebrities_tweets['text']

""" WORDCLOUDS 

wordcloud1 = WordCloud()\
    .generate(" ".join(tweet for tweet in celebrities_tweets.loc[celebrities_tweets['name'] == 'CNN', 'text']))
wordcloud2 = WordCloud()\
    .generate(" ".join(tweet for tweet in celebrities_tweets.loc[celebrities_tweets['name'] == 'Gates', 'text']))
wordcloud3 = WordCloud()\
    .generate(" ".join(tweet for tweet in celebrities_tweets.loc[celebrities_tweets['name'] == 'Trump', 'text']))
wordcloud4 = WordCloud()\
    .generate(" ".join(tweet for tweet in celebrities_tweets.loc[celebrities_tweets['name'] == 'Musk', 'text']))
wordcloud5 = WordCloud()\
    .generate(" ".join(tweet for tweet in celebrities_tweets.loc[celebrities_tweets['name'] == 'Oprah', 'text']))
wordcloud6 = WordCloud()\
    .generate(" ".join(tweet for tweet in celebrities_tweets.loc[celebrities_tweets['name'] == 'NASA', 'text']))

plt.imshow(wordcloud1, interpolation='bilinear')
plt.axis("off")
plt.title('CNN')
plt.show()
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("off")
plt.title('Gates')
plt.show()
plt.imshow(wordcloud3, interpolation='bilinear')
plt.axis("off")
plt.title('Trump')
plt.show()
plt.imshow(wordcloud4, interpolation='bilinear')
plt.axis("off")
plt.title('Musk')
plt.show()
plt.imshow(wordcloud5, interpolation='bilinear')
plt.axis("off")
plt.title('Oprah')
plt.show()
plt.imshow(wordcloud6, interpolation='bilinear')
plt.axis("off")
plt.title('NASA')
plt.show()
"""
train_inputs, test_inputs, train_classes, test_classes = train_test_split(inputs, classes,test_size=0.2, random_state=274963)

bow_transformer = CountVectorizer(analyzer=text_process).fit(train_inputs)
text_bow_train = bow_transformer.transform(train_inputs)
text_bow_test = bow_transformer.transform(test_inputs)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0])
                                  , range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Multinomial Naive Bayes Algorithm
print('-----------NAIVE BAYES-----------')
bayes = MultinomialNB()
bayes = bayes.fit(text_bow_train, train_classes)
print(bayes.score(text_bow_test, test_classes))


predictions = bayes.predict(text_bow_test)
print(classification_report(test_classes, predictions))


cm = confusion_matrix(test_classes,predictions)
plt.figure()
plot_confusion_matrix(cm, classes=['CNN', 'Gates', 'Trump', 'Musk', 'Oprah', 'NASA'], normalize=True,
                      title='Bayes Confusion Matrix')


print('-----------MLP-----------')

# MLP
mlp = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=1000, activation = 'relu',solver='adam',random_state=1)

mlp.fit(text_bow_train, train_classes)

predictions_train = mlp.predict(text_bow_train)
print(accuracy_score(predictions_train, train_classes))
predictions_test = mlp.predict(text_bow_test)
print(accuracy_score(predictions_test, test_classes))

print(classification_report(test_classes, predictions_test))


cm = confusion_matrix(test_classes, predictions_test)
plt.figure()
plot_confusion_matrix(cm, classes=['CNN', 'Gates', 'Trump', 'Musk', 'Oprah', 'NASA'], normalize=True,
                      title='MLP Confusion Matrix')


print('-----------KNN-----------')
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(text_bow_train, train_classes)
predictions_train = knn.predict(text_bow_train)
print(accuracy_score(predictions_train, train_classes))
predictions_test = knn.predict(text_bow_test)
print(accuracy_score(predictions_test, test_classes))

print(classification_report(test_classes, predictions_test))


cm = confusion_matrix(test_classes, predictions_test)
plt.figure()
plot_confusion_matrix(cm, classes=['CNN', 'Gates', 'Trump', 'Musk', 'Oprah', 'NASA'], normalize=True,
                      title='KNN Confusion Matrix')
