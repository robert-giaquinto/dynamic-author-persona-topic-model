#!/bin/bash
data_dir="/Users/robert/Local/apt/data/"
out_dir="/Users/robert/Local/apt/data/out/authors_dev/"
python -m src.Apt --input_file ${data_dir}test_apt_authors.dat --vocab_file ${data_dir}vocab.dat \
    --method inference \
    --out_dir ${out_dir} \
    --doc_topic_file train_authors_dev5_doc_topics.txt \
    --lhood_file train_authors_dev5_heldout_perword_likelihood.txt \
    --doc_lhood_file train_authors_dev5_doc_lhoods \
    --model_file train_authors_dev5.p \
    --include_penalty;
