from nltk import tokenize

class datawrapper():
    def __init__(self, dataclass):
        self._dataclass = dataclass

    def sentence_split(text):
        """
        Splits the input in sentences using NLTK library
        Args:
            text : list - List with the text to process
        Return:
            Split text as a list
        """
        #implementMP TODO
        try:
            splittext = list(map(tokenize.sent_tokenize, text))
        except TypeError:
            splittext = "ERROR: expected a list containing STR type. Were the correct tags selected?"
        return splittext

    def clean_markup(txt):
        """
        Aux function to clean HTML from a text input
        """
        from io import StringIO
        from html.parser import HTMLParser

        class Clean_markup(HTMLParser):
            def __init__(self):
                super().__init__()
                self.reset()
                self.strict = False
                self.convert_charrefs= True
                self.text = StringIO()
            def handle_data(self, d):
                self.text.write(d)
            def get_data(self):
                return self.text.getvalue()

        s = Clean_markup()
        s.feed(txt)
        return s.get_data()
    #Batchin TODO ?If you're just interested in the results, you're better off using one of multiprocessing.Pool's map functions.

if __name__ == "__main__":
    """
        TEST
    """
    print(sentence_split(list(map(clean_markup,
    ["""<br>The AllSpark is an ancient and infinitely limitless, powerful Cybertronian artifact</br>. 
    It has the power to bring lifeless technology to life by turning it into sentient, autonomous Cyberronians.
    """]
    ))))
    print(sentence_split(list(map(clean_markup,
    ["""<br>El AllSpark es un antiguo e inifinitamente ilimitado, poderoso artefacto Cybertronian</br>. 
    Tiene el poder de traer la tecnología al a vida transformándolo en un ser autónomo, Cyberronian.
    """]
    ))))

