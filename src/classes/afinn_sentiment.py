from afinn import Afinn

class AfinnSentiment():
    def __init__(self):
        self.afinn = Afinn(language='da', emoticons=True)

        self.normalize_dict = {
            -5.0: 0.0,
            -4.0: 0.1,
            -3.0: 0.2,
            -2.0: 0.3,
            -1.0: 0.4,
            0.0: 0.5,
            1.0: 0.6,
            2.0: 0.7,
            3.0: 0.8,
            4.0: 0.9,
            5.0: 1.0
        }

    def get_afinn_sentiment(self, text):
        return self.afinn.score(text)