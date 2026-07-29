import torch

from tokenizer import BPETokenizer
from model import GPT
from config import *


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)

tokenizer = BPETokenizer()
tokenizer.load("tokenizer.json")

VOCAB_SIZE = len(tokenizer.token_to_id)


model = GPT(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    context_length=CONTEXT_LENGTH,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
).to(device)


checkpoint = torch.load(
    "best_model.pt",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=1.0,
):  
    assert temperature > 0, "Temperature must be greater than 0."

    ids = tokenizer.encode(prompt)

    ids = torch.tensor(
        ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        
        for _ in range(max_new_tokens):

            input_ids = ids[:, -CONTEXT_LENGTH:]

            logits = model(input_ids)


            next_token_logits = logits[:, -1, :]

            next_token_logits = next_token_logits / temperature

            k = SAMPLE_FROM_K 

            values, indices = torch.topk(next_token_logits, k)

            filtered_logits = torch.full_like(
                next_token_logits,
                float("-inf")
            )

            filtered_logits.scatter_(
                1,
                indices,
                values
            )

            probs = torch.softmax(filtered_logits, dim=-1)

            next_token = torch.multinomial(
                probs,
                1
            )

            ids = torch.cat([ids, next_token], dim=1)

            if next_token.item() == tokenizer.token_to_id["<eos>"]:
                break

    return tokenizer.decode(ids.squeeze(0).tolist())

prompt = input("Prompt: ")

output = generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
)

print(output)