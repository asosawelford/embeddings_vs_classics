META="data/metadata.csv"
CLASSIC="data/REDLAT_features.csv"
WAVLM="/home/aleph/redlat/REDLAT_24-09-25_masked_prepro_wavlm"
ROBERTA="/home/aleph/redlat/REDLAT_24-09-25_transcriptions_gemini_roberta"
TASKS="Fugu CraftIm Phonological Phonological2 Semantic Semantic2"

# Loop over the 3 Comparison Pairs
for PAIR in "CN AD" "CN FTD" "AD FTD"; do
    # Convert string "CN AD" to array for passing to script
    set -- $PAIR
    G1=$1
    G2=$2
    
    echo "==================================================="
    echo "STARTING GROUP COMPARISON: $G1 vs $G2"
    echo "==================================================="

    # 1. Classic Audio
    echo "Running Classic Audio..."
    python src/train_paper_methodology.py --model_type classic --metadata "$META" --classic_csv "$CLASSIC" --tasks $TASKS --classic_subset audio --target_groups $G1 $G2

    # 2. Classic Language
    echo "Running Classic Language..."
    python src/train_paper_methodology.py --model_type classic --metadata "$META" --classic_csv "$CLASSIC" --tasks $TASKS --classic_subset language --target_groups $G1 $G2

    # 3. Classic Combined
    echo "Running Classic Combined..."
    python src/train_paper_methodology.py --model_type classic --metadata "$META" --classic_csv "$CLASSIC" --tasks $TASKS --classic_subset combined --target_groups $G1 $G2

    # 4. WavLM
    echo "Running WavLM..."
    python src/train_paper_methodology.py --model_type wavlm --metadata "$META" --embedding_dir "$WAVLM" --tasks $TASKS --target_groups $G1 $G2

    # 5. RoBERTa
    echo "Running RoBERTa..."
    python src/train_paper_methodology.py --model_type roberta --metadata "$META" --embedding_dir "$ROBERTA" --tasks $TASKS --target_groups $G1 $G2

done

echo "🎉 ALL EXPERIMENTS COMPLETED!"