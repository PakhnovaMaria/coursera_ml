__author__ = 'Pakhnova Maria'
from sklearn.externals import joblib
from nltk.stem.porter import PorterStemmer

class SentimentClassifier(object):
    def __init__(self):
        self.model = joblib.load("model.pkl")
        #self.stemmer = joblib.load("PorterStemmer.pkl")
        self.stemmer = PorterStemmer()
        self.classes_dict = {0: "negative", 1: "positive", -1: "error!"}
        self.color_dict = {0: "red", 1: "green", -1: "gray"}

    @staticmethod
    def get_probability_words(probability):
        if probability < 0.55:
            return "neutral or uncertain"
        if probability < 0.7:
            return "probably"
        if probability > 0.95:
            return "certain"
        else:
            return ""

    def predict_text(self, text):
        try:
            text_new = self.process_message(text)
            return self.model.predict(text_new)[0]
        except:
            return -1

    def process_message(self, text):
        text_new = []
        words = text.replace('.', '').split()
        filtered_sentence = [self.stemmer.stem(w) for w in words if w.isalpha()]
        text_new.append(" ".join(filtered_sentence))
        print(text_new)
        return text_new

    def get_prediction_message(self, text):
        class_prediction = self.predict_text(text)
        return self.classes_dict[class_prediction], self.color_dict[class_prediction]