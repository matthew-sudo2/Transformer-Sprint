"""
Train a WordPiece tokenizer for the Transformer Sprint project.

This script:
1. Initializes a WordPiece tokenizer with proper normalization
2. Trains on your text corpus to learn subword vocabulary
3. Saves the trained tokenizer for later use in model training
"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from datasets import load_dataset
import os

def main():
    # Step 1: Initialize the tokenizer with WordPiece model
    # The unk_token handles words that can't be broken down further
    print("Initializing WordPiece tokenizer...")
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

    # Step 2: Configure text normalization
    # NFKC: Unicode normalization for consistent character representation
    # Lowercase: Convert all text to lowercase for uniformity
    print("Setting up normalizers...")
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Lowercase()
    ])

    # Step 3: Set pre-tokenizer
    # Whitespace splits text into words before WordPiece applies subword splitting
    print("Configuring pre-tokenizer...")
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Step 4: Define special tokens required for transformer models
    special_tokens = [
        "[PAD]",   # Padding token for equal-length sequences
        "[UNK]",   # Unknown token for out-of-vocabulary words
        "[CLS]",   # Classification token (start of sequence)
        "[SEP]",   # Separator token (end of sequence or between sentences)
        "[MASK]"   # Mask token for masked language modeling
    ]
    print(f"Special tokens: {special_tokens}")

    # Step 5: Create the trainer with hyperparameters
    # vocab_size: Target vocabulary size (BERT uses ~30k)
    # min_frequency: Ignore tokens appearing less than this many times
    trainer = trainers.WordPieceTrainer(
        vocab_size=30000,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True
    )
    print("Trainer configured with vocab_size=30000, min_frequency=2")

    # Step 6: Load training data
    # Ensure data/train.txt exists with one sentence/paragraph per line
    data_path = "data/train.txt"
    if not os.path.exists(data_path):
        print(f"\n⚠️  Warning: {data_path} not found!")
        print("Please create this file with your training text first.")
        print("Example content:")
        print("  Welcome to the Transformer Sprint project.")
        print("  Tokenization is the first step in NLP.")
        print("  WordPiece breaks words into subwords like playing -> play, ##ing.")
        return
    
    print(f"\nLoading dataset from {data_path}...")
    dataset = load_dataset("text", data_files={"train": data_path})
    print(f"Loaded {len(dataset['train'])} lines of text")

    # Step 7: Train the tokenizer
    print("\nTraining tokenizer (this may take a moment)...")
    tokenizer.train_from_iterator(dataset["train"]["text"], trainer=trainer)
    print(f"Training complete! Vocabulary size: {tokenizer.get_vocab_size()}")

    # Step 8: Save the trained tokenizer
    output_path = "models/wordpiece_tokenizer.json"
    os.makedirs("models", exist_ok=True)
    tokenizer.save(output_path)
    print(f"\n✓ Tokenizer saved to {output_path}")
    
    # Step 9: Quick verification test
    print("\n--- Verification Test ---")
    test_text = "Tokenization is working correctly!"
    encoded = tokenizer.encode(test_text)
    print(f"Input: {test_text}")
    print(f"Tokens: {encoded.tokens}")
    print(f"IDs: {encoded.ids}")
    decoded = tokenizer.decode(encoded.ids)
    print(f"Decoded: {decoded}")

if __name__ == "__main__":
    main()
