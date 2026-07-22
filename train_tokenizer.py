from tokenizer import BPETokenizer
    
with open("input.txt", encoding="utf-8") as f:
    text = f.read()

# Load tokenizer
tokenizer = BPETokenizer()

tokenizer.train(text, num_merges=3000)