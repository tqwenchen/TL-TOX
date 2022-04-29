import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as f

class dvib(nn.Module):
    def __init__(self, k,out_channels, hidden_size):
        super(dvib, self).__init__()

        self.conv = torch.nn.Conv2d(in_channels=1,
                                    out_channels=out_channels,
                                    kernel_size=(1, 20),
                                    stride=(1, 1),
                                    padding=(0, 0),
                                    )

        self.rnn = torch.nn.GRU(input_size=out_channels,
                                hidden_size=hidden_size,
                                num_layers=2,
                                bidirectional=True,
                                batch_first=True,
                                dropout=0.2
                                )

        self.fc1 = nn.Linear(hidden_size * 4, hidden_size * 4)
        self.enc_mean = nn.Linear(hidden_size * 4 + 578, k)
        self.enc_std = nn.Linear(hidden_size * 4 + 578, k)
        self.dec = nn.Linear(k, 2)

        self.drop_layer = torch.nn.Dropout(0.5)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.enc_mean.weight)
        nn.init.constant_(self.enc_mean.bias, 0.0)
        nn.init.xavier_uniform_(self.enc_std.weight)
        nn.init.constant_(self.enc_std.bias, 0.0)
        nn.init.xavier_uniform_(self.dec.weight)
        nn.init.constant_(self.dec.bias, 0.0)

    def cnn_gru(self, x, lens):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.nn.ReLU()(x)
        x = x.squeeze(3)
        x = x.permute(0, 2, 1)

        gru_input = pack_padded_sequence(x, lens, batch_first=True)
        output, hidden = self.rnn(gru_input)

        output_all = torch.cat([hidden[-1], hidden[-2], hidden[-3], hidden[-4]], dim=1)

        return output_all

    def forward(self, pssm, lengths, FEGS):
        cnn_vectors = self.cnn_gru(pssm, lengths)
        feature_vec = torch.cat([cnn_vectors, FEGS], dim=1)

        enc_mean, enc_std = self.enc_mean(feature_vec), f.softplus(self.enc_std(feature_vec) - 5)
        eps = torch.randn_like(enc_std)
        latent = enc_mean + enc_std * eps

        outputs = torch.sigmoid(self.dec(latent))
        #         print(outputs.shape)

        return outputs, enc_mean, enc_std, latent

def rcnn1():
    return dvib(k=1024, out_channels=128, hidden_size=512)


