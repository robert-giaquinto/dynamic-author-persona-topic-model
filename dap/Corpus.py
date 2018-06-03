from __future__ import division, print_function, absolute_import
import numpy as np
from dap.Doc import Doc
import logging
import argparse
import time

logger = logging.getLogger(__name__)


class Corpus(object):
    """
    Container of corpus information

    For now, just store entire corpus in memory
    TODO: provide iterators over corpus, do one pass over file to collect meta information
    """
    def __init__(self, input_file, vocab_file, in_memory=False, author2id=None, log=True):
        if log:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        self.docs = []
        self.num_docs = 0
        self.total_words = 0
        self.times = []
        self.num_times = 0
        self.vocab_size = 0
        self.max_length = 0

        if author2id is None:
            self.author2id = {}
            self.num_authors = 0
        else:
            self.author2id = author2id
            self.num_authors = len(author2id)

        self.vocab = []
        self.input_file = input_file
        self.num_docs_per_time = []
        self.in_memory = in_memory

        # read and process the input file
        if in_memory:
            skipped_docs, skipped_times = self.read_file()
        else:
            skipped_docs, skipped_times = self.scan_file()

        # read and process the vocabulary file
        self.read_vocab(vocab_file)

        logger.info("PROCESSED CORPUS")
        logger.info("Number of time points: " + str(self.num_times))
        logger.info("Number of authors: " + str(self.num_authors))
        logger.info("Number of documents: " + str(self.num_docs))
        logger.info("Total number of words: " + str(self.total_words))
        logger.info("Found ids for " + str(self.vocab_size) + " terms in vocabulary")
        logger.info("Number of documents skipped (no words): " + str(skipped_docs))
        logger.info("Number of times skipped (no documents): " + str(skipped_times))
        logger.info("Number of documents per time-step: {}".format(
            ", ".join([str(i) + ":" + str(n) for i, n in enumerate(self.num_docs_per_time)])))

    def read_file(self):
        """
        Main processing method for reading and parsing information in the file
        """
        skipped_docs = 0
        skipped_times = 0
        doc_id = 0

        for_training = len(self.author2id) == 0 # don't include new authors in a test set
        if for_training:
            logger.info("Building training corpus.")
        else:
            logger.info("Building testing corpus. Will skip any new authors.")

        with open(self.input_file, "rb") as f:
            self.num_times = int(f.readline().replace('\n', ''))
            self.num_docs_per_time = [0] * self.num_times
            for t in range(self.num_times):
                # catch newlines at the end of the file
                line = f.readline().replace('\n', '')
                if line == '':
                    break

                time_stamp = int(float(line))
                num_docs = int(f.readline().replace('\n', ''))
                if num_docs == 0:
                    skipped_times += 1
                    continue

                self.times.append(time_stamp)
                self.num_docs += num_docs

                for d in range(num_docs):
                    doc = Doc()
                    doc.time = time_stamp
                    doc.time_id = t

                    # read one line = one document
                    fields = f.readline().replace('\n', '').split()

                    # extract author
                    doc.author = fields[0]

                    # convert author to a unique author id
                    if doc.author not in self.author2id:
                        if for_training:
                            self.author2id[doc.author] = self.num_authors
                            self.num_authors += 1
                        else:
                            # skip this doc, don't want to add new authors to a test set
                            skipped_docs += 1
                            continue

                    # save author id
                    doc.author_id = self.author2id[doc.author]

                    doc.num_terms = int(fields[1])
                    if self.max_length < doc.num_terms:
                        self.max_length = doc.num_terms

                    # extract words and corresponding counts in this document
                    word_counts = [[int(elt) for elt in wc.split(":")] for wc in fields[2:]]
                    if len(word_counts) == 0:
                        self.num_docs -= 1
                        skipped_docs += 1
                        continue

                    doc.doc_id = doc_id
                    doc_id += 1
                    self.num_docs_per_time[t] += 1

                    doc.words = np.array([w for w, c in word_counts])
                    doc.counts = np.array([c for w, c in word_counts])
                    self.docs.append(doc)
                    self.total_words += np.sum(doc.counts)

                    if max(doc.words) > self.vocab_size:
                        self.vocab_size = max(doc.words) + 1

        return skipped_docs, skipped_times

    def scan_file(self):
        """
        Simply scan over the dataset once to get meta information
        """
        self.num_authors = 0
        skipped_docs = 0
        skipped_times = 0
        with open(self.input_file, "rb") as f:
            self.num_times = int(f.readline().replace('\n', ''))
            self.num_docs_per_time = [0] * self.num_times
            for t in range(self.num_times):
                # catch newlines at the end of the file
                line = f.readline().replace('\n', '')
                if line == '':
                    break

                time_stamp = int(float(line))
                num_docs = int(f.readline().replace('\n', ''))
                if num_docs == 0:
                    skipped_times += 1
                    continue

                self.times.append(time_stamp)
                self.num_docs += num_docs

                for d in range(num_docs):
                    # read one line = one document
                    fields = f.readline().replace('\n', '').split()

                    # extract author
                    author = fields[0]

                    # convert author to a unique author id
                    if author not in self.author2id:
                        self.author2id[author] = self.num_authors
                        self.num_authors += 1

                    # save author id
                    author_id = self.author2id[author]

                    num_terms = int(fields[1])
                    if self.max_length < num_terms:
                        self.max_length = num_terms

                    # extract words and corresponding counts in this document
                    word_counts = [[int(elt) for elt in wc.split(":")] for wc in fields[2:]]
                    if len(word_counts) == 0:
                        self.num_docs -= 1
                        skipped_docs += 1
                        continue

                    self.num_docs_per_time[t] += 1
                    words = np.array([w for w, c in word_counts])
                    counts = np.array([c for w, c in word_counts])
                    self.total_words += np.sum(counts)

                    if max(words) > self.vocab_size:
                        self.vocab_size = max(words) + 1

        return skipped_docs, skipped_times

    def read_vocab(self, vocab_file):
        with open(vocab_file, "r") as f:
            self.vocab = tuple([v.replace("\n", "") for v in f.readlines()])
            self.vocab_size = len(self.vocab)
            logger.info("Number of words in vocabulary: " + str(len(self.vocab)))

    def __iter__(self):
        """
        Iterator returning each doc
        :return:
        """
        if self.in_memory:
            for doc in self.docs:
                yield doc
        else:
            doc_id = 0
            with open(self.input_file, "rb") as f:
                num_times = int(f.readline().replace('\n', ''))
                for t in range(num_times):
                    # catch newlines at the end of the file
                    line = f.readline().replace('\n', '')
                    if line == '':
                        break

                    time_stamp = int(float(line))
                    num_docs = int(f.readline().replace('\n', ''))
                    if num_docs == 0:
                        continue

                    for d in range(num_docs):
                        doc = Doc()
                        doc.time = time_stamp
                        doc.time_id = t
                        fields = f.readline().replace('\n', '').split()
                        doc.author = fields[0]
                        doc.author_id = self.author2id[doc.author]
                        doc.num_terms = int(fields[1])

                        # extract words and corresponding counts in this document
                        word_counts = [[int(elt) for elt in wc.split(":")] for wc in fields[2:]]
                        if len(word_counts) == 0:
                            continue

                        doc.doc_id = doc_id
                        doc_id += 1

                        doc.words = np.array([w for w, c in word_counts])
                        doc.counts = np.array([c for w, c in word_counts])
                        yield doc

    def __len__(self):
        return len(self.docs)

    def __str__(self):
        return "Corpus with " + str(self.num_times) + " time periods, and " + str(self.num_docs) + " total documents."


def main():
    """
    Example of how to run corpus class
    :return:
    """
    parser = argparse.ArgumentParser(
        description='Create corpus object from a file.')
    parser.add_argument('--input_file', type=str,
                        help='Path to input file.')
    parser.add_argument('--vocab_file', type=str,
                        help='Path to vocabulary file.')
    parser.add_argument('--log', dest="log", action="store_true",
                        help='Add this flag to have progress printed to log.')
    parser.set_defaults(log=True)
    args = parser.parse_args()

    print('Corpus.py')
    print(args, "\n")
    start = time.time()
    corpus = Corpus(args.input_file, args.vocab_file)
    end = time.time()
    print("\nTime to process the file:", round(end - start, 1), "seconds")

    print(corpus)
    print("len of corpus:", len(corpus))


if __name__ == "__main__":
    main()
