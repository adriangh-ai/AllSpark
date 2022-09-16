from nltk import tokenize

class datawrapper():
    def __init__(self):
        pass
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

    def word_split(text):
        """
        Splits the input into words using NLTK library
        Args:
            text : list - List with the text to process
        Return:
            Split text as a list
        """
        #implementMP TODO
        try:
            splittext = list(map(tokenize.word_tokenize, text))
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

if __name__ == "__main__":
    """
        TEST
    """
   
    print(datawrapper.sentence_split(list(map(datawrapper.clean_markup,
    ["""<br>The AllSpark is an ancient and infinitely limitless, powerful Cybertronian artifact</br>. 
    It has the power to bring lifeless technology to life by turning it into sentient, autonomous Cyberronians.
    """]
    ))))
    

    print(datawrapper.word_split(list(map(datawrapper.clean_markup,
    ["""<br>The AllSpark is an ancient and infinitely limitless, powerful Cybertronian artifact</br>. 
    It has the power to bring lifeless technology to life by turning it into sentient, autonomous Cyberronians.
    """]
    ))))