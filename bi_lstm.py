import torch
from torch import nn
class BiLSTM(nn.Module):
    
    def __init__(self,vocab_size,embed_size=300,hidden_size=64,lstm_layers=4,dropout=0.1,padding_id=0):
        super(BiLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size,padding_idx=padding_id)
        
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size,
                            num_layers=self.lstm_layers,
                            batch_first=True,bidirectional=True)
        self.linear = nn.Linear(self.hidden_size*4 , 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout)
        self.out = nn.Linear(64, 2)
        self.act = nn.Softmax(dim=-1)


    def forward(self, input_ids,labels):
        h_embedding = self.embedding(input_ids)
        h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        #print("avg_pool", avg_pool.size())
        #print("max_pool", max_pool.size())
        conc = torch.cat(( avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        act = self.act(out)
        seq_loss = torch.nn.CrossEntropyLoss(reduction="mean")(act, labels)
        return (seq_loss,act)