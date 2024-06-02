import torch
from torch import nn
import torch.nn.functional as F
# import gensim

EMBEDDING_SIZE = 100
NUM_FILTERS = 10

class CnnTextClassifier(nn.Module):
    def __init__(self, num_classes, window_sizes=(1,2,3)):
        super(CnnTextClassifier, self).__init__()
        # w2vmodel = gensim.models.KeyedVectors.load(OUTPUT_FOLDER + 'models/' + 'word2vec_500_PAD.model')
        # weights = w2vmodel.wv
        # # With pretrained embeddings
        # self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors), padding_idx=w2vmodel.wv.vocab['pad'].index)
        # Without pretrained embeddings
        # self.embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, NUM_FILTERS, [window_size, EMBEDDING_SIZE], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])

        self.fc = nn.Linear(NUM_FILTERS * len(window_sizes), num_classes)

    def forward(self, x, debug=False):
        # x = self.embedding(x) # [B, T, E]

        # print(x.shape)

        # Apply a convolution + max_pool layer for each window size
        x = torch.unsqueeze(x, 1)

        # print(x.shape)
        # exit()
        xs = []
        for conv in self.convs:
            x2 = conv(x)
            # print(x2.shape)
            x2 = torch.tanh(x2)
            x2 = torch.squeeze(x2, -1)
            x2 = F.max_pool1d(x2, x2.size(2))
            xs.append(x2)
        x = torch.cat(xs, 2)

        # FC
        x = x.view(x.size(0), -1)
        logits = self.fc(x)

        # probs = F.softmax(logits, dim = 1)

        return logits, 0


