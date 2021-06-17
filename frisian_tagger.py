from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression
import csv
import nltk
from sklearn.model_selection import train_test_split
import nltk.classify
from sklearn.metrics import classification_report
import sklearn
from sklearn.model_selection import KFold
import random
import numpy as np
import time
import pandas as pd

def main():
    print('Importing data...')
    data_train = importData('fry-ner-train.tsv')
    data_test = importData('fry-ner-test.tsv')

    #For testing the kfolds
    #testKFolds(data_train, save=True, title='Standard + +1- words. + word.isalnum')
    #exit()
    print('Testing LinearSVC')
    testOnTestSet(data_train, data_test, type='SVM')

    print('Testing naive Bayes')
    testOnTestSet(data_train, data_test, type='Bayes')
    

def importData(filename):
    '''
        Imports data from a given filename and returns a list with the data.

        @param filename the name of the file to be used for the data
        @return data a list containing sentences with data in them.
    '''
    columns = ['token', 'lemma', 'POS', 'tag']
    df = pd.read_csv(filename, sep='\t', names=columns, skip_blank_lines=False)

    data = []
    sent = []
    for row in df.values.tolist():
        if np.nan not in row:
            sent.append(row)
        else:
            data.append(sent)
            sent = []

    return data


def testKFolds(data, save=True, title=''):
    '''
        Function that will do a 10-fold cross-validation, given the data and optionally save
        it to a .csv file for further inspection.

        @param data The raw data, imported from the .tsv file
        @param save Boolean for saving the output to a file.
        @param title the title of the row in the optional file.
        @return f1_avg The f1 averaged.
    '''
    feats = dataToFeatureList(data)
    print(feats[0:5])
    tags = set([label for _, label in feats])
    print(tags)
    tags.remove('O')

    print(len(feats))
    #feats = feats[0:10000]

    f1_list = []
    a_list = []
    runtime_list = []
    for fold in split_folds(feats):
        train = [feats[i] for i in fold[0]]
        test = [feats[i] for i in fold[1]]

        old_time = time.time()
        classifier = SVM(train)
        runtime = time.time() - old_time

        f1, a = calcF1A(classifier, test, tags)

        print(f1, a, runtime)
        f1_list.append(f1)
        a_list.append(a)
        runtime_list.append(runtime)

    f1_avg = sum(f1_list)/len(f1_list)
    runtime_avg = sum(runtime_list)/len(runtime_list)
    a_avg = sum(a_list)/len(a_list)
    print(f"Average F1: {f1_avg}")
    print(f"Average run-time in s: {runtime_avg}")

    if save:
        std = np.std(f1_list, ddof=1)
        row = [title] + f1_list + [f1_avg, std, a_avg, round(runtime_avg, 2)]
        with open('Results2.csv', 'a+', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(row)

    return f1_avg


def testOnTestSet(data_train, data_test, type='SVM'):
    '''
        Tests the training data onto the test set for final review.

        @param data_train the raw data containing the training data
        @param data_test the raw test data.
        @param type the type of classifier that gets tested.

        @return none
    '''
    train = dataToFeatureList(data_train)
    test = dataToFeatureList(data_test)
    #train, test, tags = dataToTrainTest(data)
    #tags.remove('O')
    old_time = time.time()
    if type == 'SVM':
        clf = SVM(train)
    else:
        clf = naiveBayes(train)
    runtime = time.time() - old_time
    print(f"Runtime: {round(runtime,2)}s")
    tags = set([label for _, label in train])
    tags.remove('O')
    
    print(calcF1A(clf, test, tags))

def naiveBayes(train):
    '''
        Function to return a classifier based on naive Bayes.

        @param feature list for training the classifier : [({features}, 'Tag'), ..]
        @return classifier object
    '''
    classifier = nltk.classify.NaiveBayesClassifier.train(train)

    return classifier

def SVM(train):
    '''
        Function to return a classifier based on support vector machines.

        @param feature list for training the classifier : [({features}, 'Tag'), ..]
        @return classifier object
    '''
    classifier = nltk.classify.SklearnClassifier(LinearSVC(dual=False))
    classifier.train(train)

    return classifier

def split_folds(feats, folds=10):
    '''
        Splits the dataset into folds and returns the indexes of the folds.
    '''
    random.Random(1).shuffle(feats)

    kf = KFold(n_splits=folds)
    
    print("\n##### Splitting datasets...")
    return kf.split(feats)

def calcF1A(classifier, test, tags):

    # The gold standard
    y_true = [label for feat, label in test]

    #The predicted output by the classifier
    y_pred = [classifier.classify(feat) for feat, label in test]

    
    correct_list = [y_true[i] == y_pred[i] for i in range(len(y_pred))]

    #print(sklearn.metrics.f1_score(y_test, y_pred, average='weighted', labels=list(tags)))
    f1 = sklearn.metrics.f1_score(y_true, y_pred, average='weighted', labels=list(tags))
    a = len([val for val in correct_list if val == True]) / len(correct_list)
    print(classification_report(y_true, y_pred, labels=list(tags), digits=3))
    #print(classifier.most_informative_features(15))
    #print(eli5.explain_weights(classifier._clf))
    tags.add('O')
    most_informative_feature_for_class(classifier)

    #print_top10(classifier._vectorizer, classifier._clf, tags)
    #important_features(classifier._vectorizer, classifier._clf)
    #show_most_informative_features_in_list(classifier)
    return f1, a


def dataToFeatureList(data):
    '''
        Takes in the raw data and returns a feature list.
        @param data format: [(token, lemma, pos, ner), ...], ...]
        @return output [({features}, 'Tag'), ..]
    '''
    output = []
    for sent in data:
        x = sentToFeatures(sent)
        y = sentToLabels(sent)
        output += [(x[i], y[i]) for i in range(len(y))]

    return output

def most_informative_feature_for_class(clf, n=10):
    classifier = clf._clf
    vectorizer = clf._vectorizer
    labels = list(clf.labels())
    feature_names = vectorizer.get_feature_names()
    output = []
    for label in labels:
        labelid = labels.index(label)
        topn = sorted(zip(classifier.coef_[labelid], feature_names))[-n:]

        for coef, feat in topn:
            #print(label, feat, coef)
            output.append((label, feat, round(coef, 4)))
    
    output.sort(key=lambda a: a[2], reverse=False)

    print("{:2}   {:10} {:25} {:10}".format('', 'Label', 'Feature', 'Coefficient'))
    for i, tup in enumerate(output[:40], 1):
        print("{:2}   {:10} {:25} {:10}".format(i, *tup))

def getLowestCoef():
    pass

def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top10)))
        print(top10)

def dataToTrainTest(data):
    x = []
    y = []
    for sent in data:
        x += sentToFeatures(sent)
        y += sentToLabels(sent)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    train = [(x_train[i], y_train[i]) for i in range(len(x_train))]
    test = [(x_test[i], y_test[i]) for i in range(len(x_test))]

    tags = set(y)

    return train, test, tags

def sentToLabels(sent):
    '''
        Takes in a sentence and returns a list with only labels
        @param sent a sentence [(token, lemma, pos, ner), ...], ...]
        @return a label list
    '''
    return [label for _, _, _, label in sent]

def sentToFeatures(sent):
    '''
        Takes in a sentence and returns a list with only features
        @param sent a sentence [(token, lemma, pos, ner), ...], ...]
        @return a list with features
    '''
    return [wordToFeatures(sent, i) for i in range(len(sent))]

def wordToFeatures(sent, i):
    '''
        Takes in a token list and an index and returns a dictionary containing
        the features for a single token.

        @param sent a sentence [(token, lemma, pos, ner), ...], ...]
        @param i The index of the particular token 
        
        @return A dictionary containing features
    '''
    features = {}
    sent_length = len(sent)
    features.update({'sent_pos' : i})

    features.update(getSentPartFeatures("", sent[i]))

    #check for tokens to left
    if i > 0:
        features.update(getSimpleSentPartFeatures('-1', sent[i-1]))
        #features['BOS'] = False
    #    if i > 1:
    #        features.update(getSentPartFeatures('-2', sent[i-2]))
    else:
        pass
        #features.update(getNone('-1'))
        #features['BOS'] = True


    #check for tokens to right
    if i+1 < sent_length:
        features.update(getSimpleSentPartFeatures('+1', sent[i+1]))
        #features['EOS'] = False
    #    if i+2 < sent_length:
    #        features.update(getSentPartFeatures('+2', sent[i+2]))
    else:
        pass
        #features.update(getNone('+1'))
        #features['EOS'] = True

    return features

def getNone(offset):
    features = {
        'word'+offset : '',
        'lemma'+offset : '',
        'pos'+offset : ''
    }

    return features

def getSimpleSentPartFeatures(offset, sent_part):
    '''
        Simplified function that extracts the features of a part of a sentence.
        Primarily used for extracting features from juxtaposed words.

        @param offset a string indicating the offset of the main token.
        @param sent_part a piece of a sentence (token, lemma, pos)

        @return dictionary containing features. 
    '''
    word = sent_part[0]
    lemma = sent_part[1]
    pos = sent_part[2]
    features = {
        'word'+offset : word.lower(),
        'lemma'+offset : lemma,
        'pos'+offset : pos,
        'length'+offset : len(word),
        #'contains_dashes'+offset : '-' in word,
        #'begin'+offset : word[:1],
        #'end'+offset : word[-1:],
        #'bigram_begin'+offset : word[:2],
        #'bigram_end'+offset : word[-2:],
        #'is_title'+offset : word.istitle()
    }
    #features.update(getSyllables(word))

    return features

def getSentPartFeatures(offset, sent_part):
    '''
        Function that extracts the features of a part of a sentence.

        @param offset a string indicating the offset of the main token.
        @param sent_part a piece of a sentence (token, lemma, pos)

        @return dictionary containing features. 
    '''
    word = sent_part[0]
    lemma = sent_part[1]
    pos = sent_part[2]
    features = {
        'word'+offset : word,
        'length'+offset : len(word),
        'is_digit'+offset : word.isdigit(),
        'is_alnum'+offset : word.isalnum(),
        #'long_word' : len(word) > 4,
        'lemma'+offset : lemma.lower(),
        'pos'+offset : pos,
        #'pos_end'+offset : pos[-1:] ,
        #'gen_pos'+offset : generalisePOS(pos),
        'is_title'+offset : word.istitle(),
        #'both_word_lemma_upper' : word.istitle() and lemma.istitle(),
        'is_upper'+offset : word.isupper(),
        'contains_dashes'+offset : '-' in word,
        #'4gram_begin'+offset : word[:4],
        #'4gram_end'+offset : word[-4:],
        'trigram_begin'+offset : word[:3],
        'trigram_end'+offset : word[-3:],
        'bigram_begin'+offset : word[:2],
        'bigram_end'+offset : word[-2:],
        'begin'+offset : word[:1],
        'end'+offset : word[-1:],
        #'contains_numbers' : containsNumbers(word),
        'vowel_count' : vowelCount(word)
    }
    #features.update(getBigrams(word))
    #features.update(getSyllables(word))

    return features

def vowelCount(word):
    i = 0
    for char in word:
        if char not in '0123456789bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c':
            i += 1
    return i


def containsPunct(word):
    for char in word:
        if char in ':/.,-!=+<>;':
            return True
    return False

def isCaps(word, p=0.8):
    n = 0
    for char in word:
        if char.isupper():
            n+=1
    
    try:
        return n/len(word) > p
    except:
        return False


def getSyllables(word):
    syls = {}
    syl = ""
    for i in range(len(word)):
        syl += word[i]
        if word[i].lower() not in 'aeiouy'and i > 0 and word[i-1].lower() in 'yaeiou':
            syls[syl] = True
            syl = ""
    if syl != "":
        syls['syl-' + syl] = True

    return syls

def getBigrams(word):
    bigrams = {}
    for i in range(len(word)-1):
        char1 = word[i]
        char2 = word[i+1]
        bigrams['bi-'+char1+char2] = True
    return bigrams


def containsNumbers(word):
    for char in word:
        if char.isdigit():
            return True
    return False

if __name__ == '__main__':
    main()
