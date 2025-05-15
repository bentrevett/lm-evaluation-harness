MODELS=(
    "Qwen/Qwen3-0.6B"
)
TASKS=(
    "finqa_all"
    "multihiertt"
    "tatqa_all"
)
for MODEL_NAME in "${MODELS[@]}"
do
    for TASK_NAME in "${TASKS[@]}"
    do
        lm-eval \
             --model "hf" \
             --model_args "pretrained=$MODEL_NAME,parallelize=True,dtype=bfloat16" \
             --tasks "$TASK_NAME" \
             --device "cuda" \
             --batch_size "auto" \
             --apply_chat_template \
             --fewshot_as_multiturn \
             --output_path "results" \
             --log_samples \
             --write_out \
             --limit 10
     done
done
