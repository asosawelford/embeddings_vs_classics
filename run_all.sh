#!/bin/bash

# Define paths
META="data/metadata.csv"
CLASSIC="data/REDLAT_features.csv"
WAVLM="/home/aleph/redlat/REDLAT_24-09-25_masked_prepro_wavlm"
ROBERTA="/home/aleph/redlat/REDLAT_24-09-25_transcriptions_gemini_roberta"

# Define the sets of tasks to iterate over
TASK_SETS=(
    "Fugu CraftIm Phonological Phonological2 Semantic Semantic2"
)

echo "==================================================="
echo "🚀 STARTING FULL PRODUCTION EXPERIMENTS"
echo "==================================================="

# Loop over each set of tasks
for TASKS in "${TASK_SETS[@]}"; do
    echo "###################################################"
    echo "Running Production Task Set: $TASKS"
    echo "###################################################"

    # Loop over the 3 Comparison Pairs
    for PAIR in "CN AD"; do
        set -- $PAIR
        G1=$1
        G2=$2
        
        echo "---------------------------------------------------"
        echo "Running Group Comparison: $G1 vs $G2"
        echo "---------------------------------------------------"

        # 1. Classic Audio
        python src/train_paper_methodology.py --model_type classic --metadata "$META" --classic_csv "$CLASSIC" --tasks $TASKS --classic_subset audio --target_groups $G1 $G2

        # 2. Classic Language
        python src/train_paper_methodology.py --model_type classic --metadata "$META" --classic_csv "$CLASSIC" --tasks $TASKS --classic_subset language --target_groups $G1 $G2

        # # 3. Classic Combined (Concatenated)
        python src/train_paper_methodology.py --model_type classic --metadata "$META" --classic_csv "$CLASSIC" --tasks $TASKS --target_groups $G1 $G2

        # 4. Classic Fusion (GMU)
        python src/train_paper_methodology.py --model_type classic_fusion --metadata "$META" --classic_csv "$CLASSIC" --tasks $TASKS --target_groups $G1 $G2

        # 5. WavLM
        python src/train_paper_methodology.py --model_type wavlm --metadata "$META" --embedding_dir "$WAVLM" --tasks $TASKS --target_groups $G1 $G2

        # 6. RoBERTa
        python src/train_paper_methodology.py --model_type roberta --metadata "$META" --embedding_dir "$ROBERTA" --tasks $TASKS --target_groups $G1 $G2

        # 7. Embedding Fusion (GMU)
        python src/train_paper_methodology.py --model_type fusion --metadata "$META" --embedding_dir "$WAVLM" --roberta_dir "$ROBERTA" --tasks $TASKS --target_groups $G1 $G2
    done
done

echo "🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"