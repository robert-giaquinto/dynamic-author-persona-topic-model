#!/bin/bash
data_dir="/Users/robert/Local/apt/data/cv/"
topics=25
personas=15
penalty=0.25
penalty_str=$(echo "100 * $penalty / 1" | bc)
dataset=train
cv=0
model_file=${dataset}_cv${cv}_p${penalty_str}_K${topics}_P${personas}
python -m src.Apt --input_file ${data_dir}${dataset}_authors_apt_cv${cv}.dat \
    --vocab_file ${data_dir}${dataset}_authors_lda_cv${cv}-mult.dat.vocab \
    --out_dir ${data_dir} \
    --model_file ${model_file}.p \
    --method train \
    --measurement_noise 0.75 --process_noise 0.1 \
    --em_convergence 1e-3 --var_convergence 1e-4 --cg_convergence 1e-4 \
    --em_max_iter 20 --em_min_iter 10 \
    --num_topics ${topics} --num_personas ${personas} --penalty ${penalty} \
    --num_workers 3 --corpus_in_memory;
