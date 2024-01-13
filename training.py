import json
import csv
from itertools import product
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datefinder.file_handlers.DateModelList import DateModelList

import requests

# Define char_to_index and tag_to_index
char_to_index = {
                    'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4,
                    'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9,
                    'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 
                    'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 
                    'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 
                    'z': 25, '0': 26, '1': 27, '2': 28, '3': 29, 
                    '4': 30, '5': 31, '6': 32, '7': 33, '8': 34, 
                    '9': 35, ' ': 36, '.': 37, '+': 38, '/': 39, 
                    '-': 40, ',': 41, ':': 42, '\\': 43
                }
tag_to_index = {
                'O-None': 0,
                'B-B': 1, 'I-B': 2,
                'B-H': 3, 'I-H': 4,
                'B-I': 5, 'I-I': 6,
                'B-M': 7, 'I-M': 8, 
                'B-S': 9, 'I-S': 10, 
                'B-Y': 11, 'I-Y': 12, 
                'B-Z': 13, 'I-Z': 14, 
                'B-b': 15, 'I-b': 16, 
                'B-d': 17, 'I-d': 18,
                'B-f': 19, 'I-f': 20,
                'B-m': 21, 'I-m': 22,
                'B-p': 23, 'I-p': 24, 
                'B-y': 25, 'I-y': 26,
                'B-z': 27, 'I-z': 28      
            }

class DateDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        characters, tags = self.data[idx]
        return torch.tensor(characters, dtype=torch.long), torch.tensor(tags, dtype=torch.long)

class BiLSTMTagger(nn.Module):
    def __init__(self, num_characters, num_tags, embedding_dim, hidden_dim):
        super(BiLSTMTagger, self).__init__()
        self.embedding = nn.Embedding(num_characters, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
    
def generate_name():
        response = requests.get("https://api.namefake.com/english_united_kingdom")
        data = response.json()

        return data["name"].replace(" ", "_").replace(".", "")

class Trainer:
    def __init__(self, data_location, model_list, embedding_dim, hidden_dim, epochs, batch_size):
        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        print("CUDA available:", cuda_available)

        # Set device to GPU (cuda) if available, else CPU
        self.device = torch.device("cuda" if cuda_available else "cpu")

        self.data_location = data_location

        # Hyperparameters
        self.embedding_dim = embedding_dim #64
        self.hidden_dim = hidden_dim #128
        self.num_characters = len(char_to_index)  # Based on the char_to_index mapping
        self.num_tags = len(tag_to_index)         # Based on the tag_to_index mapping

        # Training Parameters
        self.epochs = epochs #5
        self.batch_size = batch_size #1

        # Initialize model
        self.model = BiLSTMTagger(self.num_characters, self.num_tags, self.embedding_dim, self.hidden_dim)
        self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # Load data from JSON file
        self.data = self.load_data(data_location)


    # Load and preprocess data from JSON file
    def load_data(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            raw_data = json.load(file)
        
        processed_data = []
        for sequence in raw_data:
            characters, tags = zip(*sequence)
            # Convert characters to indices and tags to tag indices
            character_indices = [char_to_index[char] for char in characters]
            tag_indices = [tag_to_index[tag] for tag in tags]
            processed_data.append((character_indices, tag_indices))
        
        return processed_data
    
    
    # Training loop
    def train_model(self):
        model_name = generate_name()

        print(f"Beginning training model - {model_name}")
        print(f"(Embed. Dim: {self.embedding_dim}, Hidden Dim: {self.hidden_dim}, Epochs: {self.epochs}, Batch Size: {self.batch_size})")
        self.model.train()
        dataloader = DataLoader(DateDataset(self.data), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            epoch_progress = 0
            for characters, tags in dataloader:
                characters, tags = characters.to(self.device), tags.to(self.device)
                # Forward pass
                outputs = self.model(characters)
                loss = self.criterion(outputs.view(-1, self.num_tags), tags.view(-1))

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_progress += 1
                print(f'\rEpoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f} ({epoch_progress/len(self.data)*100}%)', end='', flush=True)

        model_json = {"name": model_name, 
                      "embedding_dim": self.embedding_dim, 
                      "hidden_dim": self.hidden_dim,
                      "epochs": self.epochs,
                      "batch_size": self.batch_size,
                      "training_dataset": self.data_location}
        
        model_list.addModel(model_json)

        torch.save(self.model, f"./data/models/model_{model_name}.model")

if __name__ == '__main__':
    embedding_dim = [64, 96, 128]
    hidden_dim = [64, 128, 192]
    epochs = [5]
    batch_size = [1]
    dataset = ["./data/datasets/medium_dataset.json", "./data/datasets/large_dataset.json"]

    model_param_combinations = list(product(embedding_dim, hidden_dim, epochs, batch_size, dataset))

    model_list = DateModelList("./data/models/evaluations/model_list.json")

    for model_params in model_param_combinations:
        trainer = Trainer(model_list = model_list,
                          embedding_dim = model_params[0], 
                          hidden_dim = model_params[1], 
                          epochs = model_params[2], 
                          batch_size = model_params[3],
                          data_location = model_params[4])
        trainer.train_model()

