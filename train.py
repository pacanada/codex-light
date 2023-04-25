from dataclasses import dataclass
import math
import numpy as np
import torch.nn as nn
import torch
from model import Transformer
from utils import  decode, get_root, save_train_test_set, get_batch
torch.manual_seed(0)

@dataclass
class Config:
    block_size = 32
    n_blocks = 5
    epochs = int(1e4)
    vocab_size = 187
    embedding_dim = 12 # must be equal to head_size in this model but not in example
    batch_size=128
    evaluation_steps=100
    n_head=6
    learning_rate=0.001
    dropout=0.1
    load_model = True
    path_model = get_root() / "weights/v1.pt"
    def __post_init__(self):
        if self.embedding_dim%self.n_head!=0:
            raise ValueError(f"Embedding dimension {self.embedding_dim} should be a multiple of n_head={self.n_head}")
config = Config()




# Loading training and test
train = torch.load(get_root() / "data/train.pt")
test = torch.load(get_root() / "data/test.pt")

m = Transformer(config)

if config.load_model:
    try:
        m.load_state_dict(torch.load(config.path_model))
        m.eval()
        print(f"Model has been loaded from {config.load_model}")
    except:
        print("Could not load model")
loss_f = nn.CrossEntropyLoss()
optim = torch.optim.AdamW(m.parameters(), lr=config.learning_rate)
for i in range(config.epochs):
    X, y = get_batch(train, config.batch_size, config.block_size)
    
    out = m(X)
    optim.zero_grad()
    # work out the right dimensions for the cross entropy loss function
    loss = loss_f(out.view(config.batch_size*config.block_size,config.vocab_size), y.view(config.batch_size*config.block_size))
    loss.backward()
    optim.step()
    if i%config.evaluation_steps==0:
        with torch.no_grad():
            # This is quite expensive!!
            test_batch_size = config.batch_size #len(test) #config.batch_size
            X_test, y_test = get_batch(test, test_batch_size, config.block_size)
            out_test = m(X_test)
            loss_test = loss_f(out_test.view(test_batch_size*config.block_size,config.vocab_size), y_test.view(test_batch_size*config.block_size))
        print("i: ", i," Loss training: ", loss.detach().numpy(), " Loss test: ", loss_test.detach().numpy())
        # starting with \n
        #print(decode(m.generate_sequence(torch.tensor((0,0)), 100)))
        print(decode(m.generate_sequence(X_test[0].view(1,-1), 50, config.block_size)))

        torch.save(m.state_dict(),config.path_model)








