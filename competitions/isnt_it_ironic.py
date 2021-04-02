#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import zipfile

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import (
    BernoulliNB,
    GaussianNB,
)
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# since i don't have access to test data 
# i treated this dataset as whole data given

class Dataset:
    def __init__(self,
                 name="isnt_it_ironic.train.zip",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []

        with zipfile.ZipFile(name, "r") as dataset_file:
            with dataset_file.open(os.path.basename(name).replace(".zip", ".txt"), "r") as train_file:
                for line in train_file:
                    label, text = line.decode("utf-8").rstrip("\n").split("\t")
                    self.data.append(text)
                    self.target.append(int(label))

        self.target = np.array(self.target, np.int32)


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

def main(args):
    np.random.seed(args.seed)
    whole_data = Dataset()
    data_size = size_mb(whole_data.data)
    print(f"{len(whole_data.data)} total twits : Size {data_size:0.3f} MB")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        whole_data.data, whole_data.target, test_size=args.test_size, random_state=args.seed)

    if args.tfidf:
        print("Using TF-IDF")
        print()
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_df = 0.5,
            analyzer='word',
            ngram_range=(1, 2),
        )

    else:
        vectorizer = CountVectorizer(
            stop_words='english',
            max_df = 0.5,
            analyzer='word',
            binary=True,
        )

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    feature_names = vectorizer.get_feature_names()

    if args.use_chi2:
        print(f"Extracting {args.use_chi2} best features by chi2 test")
        print()
        ch2 = SelectKBest(chi2, k=args.use_chi2)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
        # get selected feature names
        feature_names = [ feature_names[i] for i in ch2.get_support(indices=True) ]

    feature_names = np.asarray(feature_names)

    classifiers = [
        BernoulliNB(),
        LinearSVC(max_iter=200, penalty="l2"),
        PassiveAggressiveClassifier(max_iter=200),
        RandomForestClassifier(),
        NearestCentroid(),
    ]

    for clf in classifiers:
        print('=' * 80)
        print()
        print(clf)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        score = accuracy_score(y_test, predictions)
        print(f"accuracy : {score:0.3f}")
        print()
        if args.get_top_words and hasattr(clf, 'coef_'):
            print(f"Top {args.get_top_words} keywords for ironic twits:")
            print()
            # sort by weights given than take most important features
            top = np.argsort(clf.coef_[0])[-args.get_top_words:]
            top = "\t".join(feature_names[top])
            print(f"{top}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # These arguments will be set appropriately by ReCodEx, even if you change them.
    parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
    parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    # For these and any other arguments you add, ReCodEx will keep your default value.
    parser.add_argument("--model_path", default="isnt_it_ironic.model", type=str, help="Model path")
    parser.add_argument("--test_size", default=0.2, 
                        type=lambda x: int(x) if x.isdigit() else float(x), help="Test set size")
    parser.add_argument("--get_top_words", default=False, 
                        type=lambda x: int(x), help="Get top words for each label")
    parser.add_argument("--use_chi2", default=False, 
                        type=lambda x: int(x), help="Extract most important features and use them")
    parser.add_argument("--tfidf", default=False,
                        const=True, nargs="?", type=str, help="Whether use tfidf or not")
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

