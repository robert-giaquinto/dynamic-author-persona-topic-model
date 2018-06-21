#!/bin/bash
data_dir="../data/signalmedia/test/"
out_dir="../results/signalmedia/blogs_aligned_3_30/"
topics=50
personas=25
penalty=0.25
penalty_str=$(echo "100 * $penalty / 1" | bc)
dataset=signalmedia_blogs
model_file=${dataset}_p${penalty_str}_K${topics}_P${personas}
python -m dap.Dap --train_file ${data_dir}train_signalmedia.dap \
	--test_file ${data_dir}test_signalmedia.dap \
    --vocab_file ${data_dir}signalmedia.bow.vocab \
    --out_dir ${out_dir} \
    --model_file ${model_file}.p \
    --method train \
    --evaluate_every 1 \
    --max_training_minutes 20.0 \
    --measurement_noise 0.8 --process_noise 0.2 \
    --em_convergence 1e-3 --var_convergence 1e-4 --cg_convergence 1e-4 \
    --em_max_iter 5 --em_min_iter 1 \
    --num_topics ${topics} --num_personas ${personas} --penalty ${penalty} \
    --num_workers 1 --corpus_in_memory --print_log;
