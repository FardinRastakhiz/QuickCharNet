import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, char_ngram_vocab_size, word_ngram_vocab_size, char_vocab_size,
                 word_seq_len, char_seq_len, embedding_size, l2_reg_lambda=0,
                 filter_sizes=[3, 4, 5, 6], mode=0):
        super(TextCNN, self).__init__()
        self.mode = mode
        self.word_seq_len = word_seq_len
        self.char_seq_len = char_seq_len
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes

        if mode in [4, 5]:
            self.char_embedding = nn.Embedding(char_ngram_vocab_size, embedding_size)
            torch.nn.init.uniform_(self.char_embedding.weight)
        if mode in [2, 3, 4, 5]:
            self.word_embedding = nn.Embedding(word_ngram_vocab_size, embedding_size)
            torch.nn.init.uniform_(self.word_embedding.weight)
        if mode in [1, 3, 5]:
            self.char_seq_embedding = nn.Embedding(char_vocab_size, embedding_size)
            torch.nn.init.uniform_(self.char_seq_embedding.weight)

        self.dropout_keep_prob = nn.Dropout(0.5)
        self.num_filters_total = 256 * len(filter_sizes) 
        self.conv_layers = nn.ModuleList()
        for filter_size in filter_sizes:
            conv = nn.Conv2d(1, 256, (filter_size, embedding_size))
            nn.init.trunc_normal_(conv.weight, std=0.1)
            nn.init.constant_(conv.bias, 0.1)
            self.conv_layers.append(conv)
        
        if mode in [3, 5]:
            self.fc_word = nn.Linear(len(filter_sizes) * 256, 512)
            torch.nn.init.xavier_normal_(self.fc_word.weight)
            torch.nn.init.constant_(self.fc_word.bias, 0.1)
            self.fc_char = nn.Linear(len(filter_sizes) * 256, 512)
            torch.nn.init.xavier_normal_(self.fc_char.weight)
            torch.nn.init.constant_(self.fc_char.bias, 0.1)
            # self.fc_concat = nn.Linear(1024, 512)
        # elif mode in [2, 4]:
        #     self.fc = nn.Linear(len(filter_sizes) * 256, 512)
        # elif mode == 1:
        #     self.fc = nn.Linear(len(filter_sizes) * 256, 512)

        self.fc1 = nn.Linear(1024, 512)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.constant_(self.fc1.bias, 0.1)
        self.fc2 = nn.Linear(512, 256)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.constant_(self.fc2.bias, 0.1)
        self.fc3 = nn.Linear(256, 128)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        torch.nn.init.constant_(self.fc3.bias, 0.1)
        self.fc4 = nn.Linear(128, 2)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.constant_(self.fc2.bias, 0.1)

    def forward(self, x_word=None, x_char=None, x_char_seq=None, x_char_pad_idx=None):
        pooled_x = []

        if self.mode in [4, 5]:
            x_char = self.char_embedding(x_char)
            x_char = x_char * x_char_pad_idx
        
        if self.mode in [2, 3, 4, 5]:
            x_word = self.word_embedding(x_word)

        if self.mode in [1, 3, 5]:
            x_char_seq = self.char_seq_embedding(x_char_seq)

        if self.mode in [4, 5]:
            x_char = torch.sum(x_char, dim=2)
            x_combined = x_char + x_word
            x_combined = x_combined.unsqueeze(-1)
        elif self.mode in [2, 3]:
            x_combined = x_word.unsqueeze(-1)
        elif self.mode in [1, 3, 5]:
            char_x_expanded = x_char_seq.unsqueeze(-1)

        if self.mode == 2 or self.mode == 3 or self.mode == 4 or self.mode == 5: 

            for i, conv in enumerate(self.conv_layers):
                h = F.relu(conv(x_combined))
                pooled = F.max_pool2d(h, (self.word_seq_len - self.filter_sizes[i] + 1, 1))
                pooled_x.append(pooled)

            h_pooled = torch.cat(pooled_x, 1) #?
            h_pooled = h_pooled.squeeze(2) #?
            h_pooled = h_pooled.squeeze(2) #?
            
            x_flat = torch.reshape(h_pooled, [-1, self.num_filters_total])  
            h_drop = self.dropout_keep_prob(x_flat) 
        
        
        if self.mode == 1 or self.mode == 3 or self.mode == 5: 
            pooled_char_x = []
            for i, conv in enumerate(self.conv_layers):
                h = F.relu(conv(char_x_expanded))
                pooled = F.max_pool2d(h, (self.char_seq_len - self.filter_sizes[i] + 1, 1))
                pooled_char_x.append(pooled)
            
            h_char_pool = torch.cat(pooled_x, 1) #?
            h_char_pool = h_pooled.squeeze(2) #?
            h_char_pool = h_pooled.squeeze(2) #?
            
            char_x_flat = torch.reshape(h_char_pool, [-1, self.num_filters_total])  
            char_h_drop = self.dropout_keep_prob(char_x_flat)
            
        # if self.mode in [3, 5]:
        #     word_output = F.relu(self.fc_word(self.dropout(h_pooled)))
        #     char_output = F.relu(self.fc_char(self.dropout(h_pooled)))
        #     conv_output = torch.cat([word_output, char_output], 1)
        if self.mode in [3, 5]:
            word_output = self.fc_word(h_drop)
            char_output = self.fc_char(char_h_drop)
            conv_output = torch.cat([word_output, char_output], 1)
        elif self.mode in [2, 4]:
            conv_output = h_drop 
        elif self.mode == 1: 
            conv_output = char_h_drop
        # else:
        #     conv_output = F.relu(self.fc(self.dropout(h_pooled)))

        output0 = F.relu(self.fc1(conv_output))
        output1 = F.relu(self.fc2(output0))
        output2 = F.relu(self.fc3(output1))
        scores = self.fc4(output2)

        return scores

# Example instantiation
model = TextCNN(char_ngram_vocab_size=100, word_ngram_vocab_size=100, char_vocab_size=100,
                word_seq_len=100, char_seq_len=100, embedding_size=128, mode=3)
print(model)
