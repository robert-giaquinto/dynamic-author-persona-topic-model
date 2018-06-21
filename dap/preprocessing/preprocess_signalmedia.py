from __future__ import division, print_function, absolute_import
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

from dap.preprocessing.utilities import stopwords, preprocessing_regex, get_wordnet_pos


# load globals used in preprocessing
lemmatizer = WordNetLemmatizer()
sw = stopwords()
stopword_set = sw.words
regex = preprocessing_regex()


def keys_generator(fname, rel_date=False):
    """
    returns generator of (source, date) pairs
    """
    with open(fname, "r") as f:
        for line in f:
            fields = line.replace("\n", "").split("\t")
            if rel_date:
                t = int(fields[1])
            else:
                t = dt.datetime.strptime(fields[1], "%Y-%m-%d %H:%M:%S")
            yield fields[0], t


def doc_generator(fname):
    with open(fname, "r") as f:
        for line in f:
            fields = line.replace("\n", "").split("\t")
            doc = fields[-1]
            yield doc.split()


def text2corpus(input_fname, corpus_fname, keep_n=25000):
    # define the the vocabulary
    docs = doc_generator(input_fname)
    vocab = corpora.Dictionary(docs)
    vocab.filter_extremes(no_above=0.5, keep_n=keep_n)
    vocab.compactify()

    # create BOW corpus
    corpus = (vocab.doc2bow(tokens) for tokens in doc_generator(input_fname))
    corpora.BleiCorpus.save_corpus(corpus_fname, corpus, id2word=vocab)


def compute_relative_dates(fname):
    keys = list(keys_generator(fname))
    # identify first date for each src
    src_min = {}
    overall_min = dt.datetime.today()
    for src, d in keys:
        if d < overall_min:
            overall_min = d

        if src not in src_min:
            src_min[src] = d
        else:
            if src_min[src] > d:
                src_min[src] = d

    # compute relative dates
    rel_dts = []
    for src, d in keys:
        days_dif = int(round((d - overall_min).days + ((d - overall_min).seconds / 60. / 60. / 24.)))
        rel_dts.append(days_dif)

    # compute counts per time step
    time_counts = Counter(rel_dts)

    # replace the current date in the data with the relative date.
    tmp_file = "tmp.txt"
    os.rename(fname, tmp_file)
    with open(tmp_file, "r") as fin, open(fname, "w") as fout:
        for line, rel_dt in zip(fin, rel_dts):
            fields = line.replace("\n", "").split("\t")
            arr = [fields[0], rel_dt, fields[2]]
            fout.write('\t'.join([str(s) for s in arr]) + "\n")
    os.remove(tmp_file)
    return time_counts


def dappify(time_counts, keys_fname, corpus_fname, full_fname):
    num_timesteps = len(time_counts)
    lda_file = open(corpus_fname, "r")
    prev_ts = -1
    with open(full_fname, "w") as fout:
        fout.write(str(num_timesteps) + "\n")
        for src, ts in keys_generator(keys_fname, rel_date=True):
            if ts != prev_ts:
                # new time step
                fout.write(str(ts) + "\n")
                fout.write(str(time_counts[ts]) + "\n")

            # otherwise, write out next source + doc
            ldac = lda_file.readline()
            fout.write(src + " " + ldac)
            prev_ts = ts

    lda_file.close()


def split_train_test(full_fname, train_fname, test_fname, test_ratio=0.0):
    # split DAP file into training and test sets
    with open(full_fname, "r") as dap_file, \
            open(train_fname, "w") as train, \
            open(test_fname, "w") as test:

        num_timesteps = int(dap_file.readline().replace("\n", ""))
        train.write(str(num_timesteps) + "\n")
        test.write(str(num_timesteps) + "\n")

        for t in range(num_timesteps):
            ts = int(dap_file.readline().replace("\n", ""))
            num_docs_t = int(dap_file.readline().replace("\n", ""))
            print("t:", t, "total:", num_docs_t)

            test_ids = np.random.choice(num_docs_t, size=int(np.ceil(num_docs_t * test_ratio)), replace=False)
            train_ids = np.delete(np.arange(num_docs_t), test_ids)

            train.write(str(ts) + "\n")
            test.write(str(ts) + "\n")
            train.write(str(len(train_ids)) + "\n")
            test.write(str(len(test_ids)) + "\n")

            for i in range(num_docs_t):
                doc = dap_file.readline()
                if i in test_ids:
                    test.write(doc)
                else:
                    train.write(doc)


def filter_sources(keep_sources, input_fname, filtered_fname, min_date=dt.datetime(year=2015,month=9, day=1)):
    # replace the current date in the data with the relative date.
    num_docs = 0
    with open(input_fname, "r") as fin, open(filtered_fname, "w") as fout:
        for line in fin:
            fields = line.replace("\n", "").split("\t")
            if fields[0] in keep_sources and dt.datetime.strptime(fields[1], "%Y-%m-%d %H:%M:%S") >= min_date:
                fout.write(line)
                num_docs += 1

    return num_docs


def check_for_news(json_dict):
    if json_dict["media-type"] == "News":
        is_news = True
    else:
        is_news = False

    return is_news


def extract_keys(json_dict):
    """
    extract keys from the json data
    :param json_dict:
    :return:
    """
    src = re.sub(r"\s+", "_", json_dict['source'])
    src = re.sub(r"[^a-zA-Z0-9_]", "_", src)
    src = re.sub(r"[_]+", "_", src)
    src = re.sub(r'^([0-9])', r'_\1', src)
    published_date = dt.datetime.strptime(json_dict['published'], "%Y-%m-%dT%H:%M:%SZ")
    # keys = [src, published_date, json_dict["media-type"]]
    keys = [src, published_date]
    return keys


def extract_text(json_dict):
    """
    Pull out the raw text from the dictionary and remove newlines
    :param json_dict:
    :return:
    """
    text = ' '.join([json_dict['title'], json_dict['content']]).strip()
    text = re.sub(r"\s+", ' ', text)
    return text


def scrub_text(text):
    """
        Defines how to clean each of the texts
        :param text:
        :return:
        """
    # all to lowercase
    text = text.lower()

    text = regex.html.sub(" ", text)
    text = regex.years.sub(" _year_ ", text)
    text = regex.dollars.sub(" _dollars_ ", text)
    text = regex.percent.sub(" _percent_ ", text)
    text = regex.times.sub(" _time_ ", text)
    text = regex.urls.sub(" _url_ ", text)
    text = regex.dates.sub(" _date_ ", text)
    text = regex.special_chars.sub(" ", text)
    text = regex.emails.sub(" _email_ ", text)

    # treat hyphens between words as a single word
    text = re.sub(r"([a-zA-Z])\-([a-zA-Z])", r"\1_\2", text)

    # expand contractions
    text = regex.iam.sub("i am", text)
    text = regex.ive.sub("i have", text)
    text = regex.hes.sub("he is", text)
    text = regex.shes.sub("she is", text)
    text = regex.weve.sub("we have", text)
    text = regex.youve.sub("you have", text)
    text = regex.willnot.sub("will not", text)
    text = regex.cannot.sub("can not", text)
    text = regex.itis.sub("it is", text)
    text = regex.letus.sub("let us", text)
    text = regex.heis.sub("he is", text)
    text = regex.sheis.sub("she is", text)
    text = regex.howis.sub("how is", text)
    text = regex.thatis.sub("that is", text)
    text = regex.thereis.sub("there is", text)
    text = regex.whatis.sub("what is", text)
    text = regex.whereis.sub("where is", text)
    text = regex.whenis.sub("when is", text)
    text = regex.whois.sub("who is", text)
    text = regex.whyis.sub("why is", text)
    text = regex.youall.sub("you all", text)
    text = regex.youare.sub("you are", text)
    text = regex.would.sub(" would", text)
    text = regex.will.sub(" will", text)
    text = regex.s_apostrophe.sub("s has ", text)
    text = regex.has.sub(" has", text)
    text = regex.nt.sub(" not", text)
    text = regex.have.sub(" have", text)

    # remove punctuation
    text = regex.punct.sub(" ", text)

    # tokenize and lemmatize
    text = [lemmatizer.lemmatize(w, pos=get_wordnet_pos(p)).lower() \
                if get_wordnet_pos(p) != '' \
                else lemmatizer.lemmatize(w).lower() \
            for w, p in pos_tag(text.split())]

    # remove stopwords
    text = [w for w in text if w not in stopword_set]
    return ' '.join(text)


def parse_json(input, output):
    sources = Counter()
    i = 0
    with open(input, 'r') as fin, open(output, "w") as fout:
        for line in fin:
            # parse the json into a dictionary
            json_dict = json.loads(line)

            is_news = check_for_news(json_dict)
            if is_news:
                continue

            i += 1
            # pull out the data we need from the text
            a_key = extract_keys(json_dict)
            a_text = extract_text(json_dict)
            sources.update([a_key[0]])

            # clean text
            clean_text = scrub_text(a_text)

            fout.write('\t'.join([str(s) for s in a_key]) + "\t" + clean_text + "\n")

    return sources


def reload_source_keys(fname):
    """
    Loop through the already preprocessed text and pull out the keys
    associated with each news source
    :param fname: name of the keys_and_text file
    """
    sources = Counter()
    with open(fname, 'r') as fin:
        for line in fin:
            fields = line.replace("\n", "").split("\t")
            sources.update([fields[0]])
    return sources


def main():
    parser = argparse.ArgumentParser(description='Preprocess signalmedia data.')
    parser.add_argument('--min_threshold', type=int, default=3)
    parser.add_argument('--max_threshold', type=int, default=30)
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of where to save DAP data.")
    parser.add_argument('--input_file', type=str, help="full file path to the signalmedia-m.jsonl file.")
    parser.add_argument('--keys_and_text_file', type=str, default="signalmedia_keys_and_text.txt")
    parser.add_argument('--filtered_file', type=str, default="filtered.txt")
    parser.add_argument('--corpus_file', type=str, default="signalmedia.bow")
    parser.add_argument('--dap_file', type=str, default="signalmedia.dap")
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--vocab_size', type=int, default=25000)
    args = parser.parse_args()

    keys_and_text_fname = os.path.join(args.data_dir, args.keys_and_text_file)
    filtered_fname = os.path.join(args.data_dir, args.filtered_file)
    corpus_fname = os.path.join(args.data_dir, args.corpus_file)
    dap_all_fname = os.path.join(args.data_dir, args.dap_file)
    train_fname = os.path.join(args.data_dir, "train_" + args.dap_file)
    test_fname = os.path.join(args.data_dir, "test_" + args.dap_file)

    path_to_current_file = os.path.abspath(os.path.dirname(__file__))
    if args.input_file is None:
        input_file = os.path.join(path_to_current_file, "../../data/signalmedia/signalmedia-1m.jsonl")
    else:
        input_file = args.input_file

    # check if the raw data exists
    if not os.path.isfile(input_file):
        raise ValueError("Raw Signal Media 1M file not found. See: http://research.signalmedia.co/newsir16/signal-dataset.html to download the raw data." )

    # parse out the keys and save cleaned up text to keys_and_text
    rerun = False
    if os.path.isfile(keys_and_text_fname) and rerun == False:
        # don't reprocess all the text, just use previous version and reload the list of new sources
        sources = reload_source_keys(keys_and_text_fname)
    else:
        # reprocess the whole text file, save text to keys_and_text
        sources = parse_json(input_file, keys_and_text_fname)

    # save only the news sources with at least min_threshold articles, but no more than max_threshold
    keep_sources = set([src for src in sources if args.min_threshold <= sources[src] <= args.max_threshold])
    num_docs = filter_sources(keep_sources, keys_and_text_fname, filtered_fname,
                              min_date=dt.datetime(year=2015, month=9, day=1))
    print("Keeping {} documents in the corpus and {} sources".format(num_docs, len(keep_sources)))

    # transform the dates into relative dates
    time_counts = compute_relative_dates(filtered_fname)
    print("time counts", time_counts)

    # sort the filtered data by source and date
    # TODO: use python to sort data in filtered_fname instead of bash
    cmd = """/bin/bash -c "sort %s -n -t $'\t' -k1,1 -k2,2 -o %s -S %s" """ % (filtered_fname, filtered_fname, "75%")
    subprocess.call(cmd, shell=True)

    # convert texts to bag-of-words format
    text2corpus(filtered_fname, corpus_fname, keep_n=args.vocab_size)

    # arrange data into DAP format
    dappify(time_counts, filtered_fname, corpus_fname, dap_all_fname)
    os.remove(filtered_fname)
    os.remove(corpus_fname)

    # split train test
    split_train_test(dap_all_fname, train_fname, test_fname, test_ratio=args.test_ratio)
    os.remove(dap_all_fname)


if __name__ == "__main__":
    main()
