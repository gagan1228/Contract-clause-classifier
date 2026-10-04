import sys
import sklearn
import torch
import transformers
from transformers import AutoTokenizer

def main() -> None:
    print(f"Python       : {sys.version.split()[0]}")
    print(f"torch        : {torch.__version__}")
    print(f"transformers : {transformers.__version__}")
    print(f"scikit-learn : {sklearn.__version__}")
    print(f"CUDA (GPU)   : {torch.cuda.is_available()}")

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    text = "This Agreement shall be governed by the laws of Delaware."
    enc = tok(text)
    print("Tokens:", tok.convert_ids_to_tokens(enc["input_ids"]))
    print("IDs   :", enc["input_ids"])


if __name__ == "__main__":
    main()