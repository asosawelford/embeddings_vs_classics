#!/bin/bash

# Define paths
META="data/metadata.csv"
CLASSIC="data/REDLAT_features.csv"
WAVLM="/home/aleph/redlat/REDLAT_24-09-25_masked_prepro_wavlm"
ROBERTA="/home/aleph/redlat/REDLAT_24-09-25_transcriptions_gemini_roberta"
TASKS="Fugu CraftIm Phonological Phonological2 Semantic Semantic2"

echo "==================================================="
echo "🔎 STARTING DEBUG SUITE (ALL MODELS / ALL PAIRS)"
echo "==================================================="

for PAIR in "CN AD" "CN FTD" "AD FTD"; do
    set -- $PAIR
    G1=$1
    G2=$2
    echo "--- Testing Pair: $G1 vs $G2 ---"

    # 1. Classic Audio
    echo " > Classic Audio..."
    python src/train_paper_methodology.py --model_type classic --metadata "$META" --classic_csv "$CLASSIC" --tasks $TASKS --classic_subset audio --target_groups $G1 $G2 --quick_debug || exit 1

    # 2. Classic Language
    echo " > Classic Language..."
    python src/train_paper_methodology.py --model_type classic --metadata "$META" --classic_csv "$CLASSIC" --tasks $TASKS --classic_subset language --target_groups $G1 $G2 --quick_debug || exit 1

    # 3. Classic Combined (No Subset Flag)
    echo " > Classic Combined..."
    python src/train_paper_methodology.py --model_type classic --metadata "$META" --classic_csv "$CLASSIC" --tasks $TASKS --target_groups $G1 $G2 --quick_debug || exit 1

    # 4. WavLM
    echo " > WavLM..."
    python src/train_paper_methodology.py --model_type wavlm --metadata "$META" --embedding_dir "$WAVLM" --tasks $TASKS --target_groups $G1 $G2 --quick_debug || exit 1

    # 5. RoBERTa
    echo " > RoBERTa..."
    python src/train_paper_methodology.py --model_type roberta --metadata "$META" --embedding_dir "$ROBERTA" --tasks $TASKS --target_groups $G1 $G2 --quick_debug || exit 1

    # 6. Fusion (GMU)
    echo " > Fusion (GMU)..."
    python src/train_paper_methodology.py --model_type fusion --metadata "$META" --embedding_dir "$WAVLM" --roberta_dir "$ROBERTA" --tasks $TASKS --target_groups $G1 $G2 --quick_debug || exit 1

done

echo "ALL DEBUG TESTS PASSED! YOU ARE READY FOR PRODUCTION."