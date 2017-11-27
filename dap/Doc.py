from __future__ import division, print_function, absolute_import


class Doc(object):
    """
    container for each document
    """
    def __init__(self):
        self.num_terms = 0
        self.words = None  # converted to np array after running corpus
        self.counts = None  # converted to np array after running corpus
        self.author = None
        self.author_id = None
        self.time = None
        self.time_id = None
        self.doc_id = None

    def __str__(self):
        return "num_terms: {}\nlen words: {}\nauthor: {}\nauthor id: {}\ntime: {}\ntime id: {}\ndoc id: {}\n".format(
            self.num_terms, len(self.words), self.author, self.author_id, self.time, self.time_id, self.doc_id)