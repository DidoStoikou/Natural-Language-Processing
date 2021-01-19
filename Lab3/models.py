import torch

from torch import nn
import numpy as np
from utils.load_embeddings import load_word_vectors

class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__()

        # 1 - define the embedding layer
        # 2 - initialize the weights of our Embedding layer
        # 3 - define if the embedding layer will be frozen or finetuned
        # from the pretrained word embeddings
        self.embeddings = nn.Embedding.from_pretrained(torch.from_numpy(embeddings),freeze = True ) # EX4

        # 4 - define a non-linear transformation of the representations
        self.hidden_size = 128
        self.linear1 = nn.Linear(embeddings.shape[1], self.hidden_size)
        self.relu = nn.ReLU()  # EX5

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        self.linear2 = nn.Linear(self.hidden_size, output_size)  # EX5

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = len(x)

        # 1 - embed the words, using the embedding layer
        embeddings = self.embeddings(x)  # EX6

        # 2 - construct a sentence representation out of the word embeddings
        representations = torch.zeros([batch_size, embeddings.shape[2]]).to(DEVICE)
        for i in range(batch_size):
            representations[i] = torch.sum(embeddings[i], dim=0) / lengths[i] # EX6

        # 3 - transform the representations to new ones.
        representations = self.relu(self.linear1(representations))  # EX6

        # 4 - project the representations to classes using a linear layer
        logits = self.linear2(representations) # EX6

        return logits
