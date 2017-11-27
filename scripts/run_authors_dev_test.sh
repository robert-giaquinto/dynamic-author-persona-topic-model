#!/bin/bash
data_dir="/Users/robert/Local/apt/data/"
out_dir="/Users/robert/Local/apt/data/out/authors_dev/"
python -m src.Apt --input_file ${data_dir}test_apt_authors.dat --vocab_file ${data_dir}vocab.dat --out_dir ${out_dir} \
       --model_file train_authors_dev5.p --method train --lag 20 \
       --measurement_noise 0.75 --process_noise 0.1 \
       --em_convergence 1e-3 --var_convergence 1e-4 --cg_convergence 1e-4 \
       --em_max_iter 12 --num_topics 15 --num_personas 4 --penalty 0.03 \
       --num_workers 3;
