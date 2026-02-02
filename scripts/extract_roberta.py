"""
Use in CL like this:
python scripts/extract_roberta_spanish.py --input_dir "path/to/transcripts" --output_dir "path/to/save/text_embeddings"
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
from transformers import AutoTokenizer, AutoModel

def main(args):
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use xlm-roberta-base for Spanish (and other languages) support
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # We do NOT need output_hidden_states=True anymore
    model = AutoModel.from_pretrained(args.model_name).to(DEVICE)
    model.eval()

    # Find all text files
    text_files = list(input_dir.rglob("*.txt"))
    print(f"Found {len(text_files)} transcript files.")
    
    if not text_files:
        print("No .txt files found.")
        return

    for text_path in tqdm(text_files, desc="Extracting final text embeddings"):
        try:
            # --- Setup Output Path ---
            relative_path = text_path.relative_to(input_dir)
            output_filename = (output_dir / relative_path).with_suffix('.npz')
            output_filename.parent.mkdir(parents=True, exist_ok=True)

            # --- Read Text ---
            with open(text_path, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()

            if not text_content:
                continue

            # --- Tokenize ---
            inputs = tokenizer(
                text_content, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            input_ids = inputs.input_ids.to(DEVICE)
            attention_mask = inputs.attention_mask.to(DEVICE)

            # --- Forward Pass ---
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)

            # --- Extract Final Layer Only ---
            # last_hidden_state shape: (Batch=1, Seq_Len, Hidden=768)
            last_hidden_state = outputs.last_hidden_state

            # --- Mean Pooling Strategy ---
            # We average all token vectors to get one sentence vector.
            # We must ignore padding tokens in this average.
            
            # 1. Expand mask to match embedding dimensions
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            
            # 2. Zero out padding embeddings
            masked_embeddings = last_hidden_state * mask_expanded
            
            # 3. Sum valid embeddings
            summed_embeddings = torch.sum(masked_embeddings, dim=1)
            
            # 4. Count valid tokens
            summed_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            
            # 5. Average
            mean_pooled = summed_embeddings / summed_mask # Shape: (1, 768)
            
            # 6. Convert to numpy vector (768,)
            final_vector = mean_pooled.squeeze(0).cpu().numpy()

            # --- Save ---
            # We save it as 'embedding' (singular) to denote it's just one layer
            np.savez_compressed(output_filename, embedding=final_vector)

        except Exception as e:
            print(f"Error processing {text_path.name}: {e}")
            continue

    print("\nText extraction complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing .txt files")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save .npz files")
    
    # Default changed to xlm-roberta-base for Spanish support
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base", help="HuggingFace model name")
    
    args = parser.parse_args()
    main(args)