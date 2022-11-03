from joblib import load

class Model:

    def __init__(self,columns):
        self.model = load("modelo.joblib")

    def make_predictions(self, data):
        result =list()
        for x in data:
            result.append(self.model.predict(x))

        return result
