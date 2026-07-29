from tokenizer import BPETokenizer

with open("input.txt", encoding="utf-8") as f:
    text = f.read()

tokenizer = BPETokenizer()
tokenizer.train(text, num_merges=3000)
tokenizer.save("tokenizer.json")
print("Tokenizer saved to tokenizer.json")