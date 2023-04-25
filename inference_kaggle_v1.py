from model import Transformer
from utils import get_root, decode, encode
from dataclasses import dataclass
import torch
#torch.manual_seed(1)
@dataclass
class Config:
    block_size = 200
    n_blocks = 6
    epochs = int(1e4)
    vocab_size = 187
    embedding_dim = 48 # must be equal to head_size in this model but not in example
    batch_size=256
    evaluation_steps=300
    n_head=6
    learning_rate=0.0003
    dropout=0.1
    load_model = True
    path_model = get_root() / "weights/v1_kaggle.pt"
    def __post_init__(self):
        if self.embedding_dim%self.n_head!=0:
            raise ValueError(f"Embedding dimension {self.embedding_dim} should be a multiple of n_head={self.n_head}")
config = Config()
test = torch.load(get_root() / "data/test.pt")

m = Transformer(config)

if config.load_model:
    try:
        m.load_state_dict(torch.load(config.path_model, map_location=torch.device('cpu')))
        m.eval()
        print(f"Model has been loaded from {config.load_model}")
    except:
        print("Could not load model")

input = test[:config.block_size]
#input_ini = "MAIDER:\n\n Oh my lord!\n\nPABLO:\n\n"
#input = (config.block_size-len(input_ini)-1)*" " + "\n" + input_ini
#input = torch.tensor(data=encode(input), dtype=torch.long)

print("INPUT")
print("------------------")
print(decode(input))
print("\nGENERATED")
print("------------------")
print(decode(m.generate_sequence(input.view(1,-1), 300, config.block_size))[config.block_size:])