import re
import os
import json
import datetime as dt
from collections import Counter
from gensim import corpora
import numpy as np
import subprocess
import argparse

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


class stopwords(object):
    """
    Simple container of stopwords. Mostly all from the NLTK set, but with a few more added on.
    """
    def __init__(self, additional_words=[]):
        if type(additional_words) is str:
            additional_words = list(additional_words)
        self.words = set(["a", "all", "am", "an",
            "and", "any", "are", "as", "at", "be", "because", "been",
            "being", "but", "by", "can",
            "cannot", "could", "did", "do", "does", "doing", "down",
            "each", "few", "for", "had", "has", "have", "having",
            "here", "if", "in", "into", "is", "it", "its", "itself", "me", "more",
            "most", "nor", "not", "of", "off", "on",
            "only", "or", "other", "ought", "ourselves", "out", "over",
            "own", "same", "should", "so", "some", "such", "than", "that", "the",
            "then", "there", "these",
            "this", "those", "through", "to", "too", "under", "until", "up", "very",
            "was", "were", "what", "when", "where", "which", "while", "who",
            "whom", "why", "with", "would", "return", "arent", "cant", "couldnt", "didnt", "doesnt",
            "dont", "hadnt", "hasnt", "havent", "hes", "heres", "hows", "im", "isnt",
            "its", "lets", "mustnt", "shant", "shes", "shouldnt", "thats", "theres",
            "theyll", "theyre", "theyve", "wasnt", "were", "werent", "whats", "whens",
            "wheres", "whos", "whys", "wont", "wouldnt", "youd", "youll", "youre",
            "youve", "will", "came", "though",
            "way", "come", "might", "now", "much",
            "i", "he", "she", "we", "they", "you", "say", "their", "his", "her", "your", "him", "ve", "re",
            "think", "thing", "about", "tell", "many", "give", "before", "after", "my", "start", "end",
            "go", "about", "make", "get", "also", "our", "them"] +
            additional_words +
            list("abcdefghjklmnopqrstuvwxyz"))


class preprocessing_regex(object):
    """
    Simple container to hold all the compiled regexes
    """
    def __init__(self):
        self.iam = re.compile(r"\bi'm\b", re.IGNORECASE)
        self.ive = re.compile(r"\bive\b", re.IGNORECASE)
        self.hes = re.compile(r"\bhes\b", re.IGNORECASE)
        self.shes = re.compile(r"\bshes\b", re.IGNORECASE)
        self.weve = re.compile(r"\bweve\b", re.IGNORECASE)
        self.youve = re.compile(r"\byouve\b", re.IGNORECASE)
        self.willnot = re.compile(r"\bwon't\b", re.IGNORECASE)
        self.cannot = re.compile(r"\bcan't\b", re.IGNORECASE)
        self.itis = re.compile(r"\bit's\b", re.IGNORECASE)
        self.letus = re.compile(r"\blet's\b", re.IGNORECASE)
        self.heis = re.compile(r"\bhe's\b", re.IGNORECASE)
        self.sheis = re.compile(r"\bshe's\b", re.IGNORECASE)
        self.howis = re.compile(r"\bhow's\b", re.IGNORECASE)
        self.thatis = re.compile(r"\bthat's\b", re.IGNORECASE)
        self.thereis = re.compile(r"\bthere's\b", re.IGNORECASE)
        self.whatis = re.compile(r"\bwhat's\b", re.IGNORECASE)
        self.whereis = re.compile(r"\bwhere's\b", re.IGNORECASE)
        self.whenis = re.compile(r"\bbwhen's\b", re.IGNORECASE)
        self.whois = re.compile(r"\bwho's\b", re.IGNORECASE)
        self.whyis = re.compile(r"\bwhy's\b", re.IGNORECASE)
        self.youall = re.compile(r"y'all|ya'll", re.IGNORECASE)
        self.youare = re.compile(r"\byou're\b", re.IGNORECASE)
        self.would = re.compile(r"'d\b", re.IGNORECASE)
        self.has = re.compile(r"'s\b", re.IGNORECASE)
        self.nt = re.compile(r"n't\b", re.IGNORECASE)
        self.will = re.compile(r"'ll\b", re.IGNORECASE)
        self.have = re.compile(r"'ve\b", re.IGNORECASE)
        self.s_apostrophe = re.compile(r"s'\b", re.IGNORECASE)
        self.punct = re.compile(r"[^a-zA-Z_ ]")
        self.special_chars = re.compile(r"&[a-z]+;")
        self.urls = re.compile(r'((http|ftp|https)://)?([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',
                          re.IGNORECASE)
        self.times = re.compile(r"(2[0-3]|[01]?[0-9]):([0-5]?[0-9])")
        self.dates = re.compile(r"\b1?[0-9]?[-/]1?[0-9]?[-/](18|19|20)?[0-9]{2}\b")
        self.percent = re.compile(r"[1-9][0-9\.]*\%")
        self.dollars = re.compile(r"\$[1-9][0-9,\.]*")
        self.years = re.compile(r"\b(18|19|20)[0-9]{2}\b")
        self.html = re.compile(r"<[^>]*>")
        self.emails = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}\b", re.IGNORECASE)


def get_wordnet_pos(treebank_tag):
    """
    Need to know the part of speech of each word to properly lemmatize.
    this function standardizes the POS codes so that they're understandable
    by the lemmatizing function.
    :param treebank_tag:
    :return:
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
