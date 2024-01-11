import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

# Set device to GPU (cuda) if available, else CPU
device = torch.device("cuda" if cuda_available else "cpu")

data_location = 'dataset.json'

# Load and preprocess data from JSON file
def load_data(filename):
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

# Define char_to_index and tag_to_index based on your dataset
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
                'B-B': 1, 
                'I-B': 2,
                'B-H': 3, 
                'I-H': 4,
                'B-I': 5, 
                'I-I': 6,
                'B-M': 7, 
                'I-M': 8, 
                'B-S': 9, 
                'I-S': 10, 
                'B-Y': 11, 
                'I-Y': 12, 
                'B-Z': 13, 
                'I-Z': 14, 
                'B-b': 15, 
                'I-b': 16, 
                'B-d': 17, 
                'I-d': 18,
                'B-f': 19, 
                'I-f': 20,
                'B-m': 21, 
                'I-m': 22,
                'B-p': 23, 
                'I-p': 24, 
                'B-y': 25, 
                'I-y': 26,
                'B-z': 27, 
                'I-z': 28      
            }

# Hyperparameters
embedding_dim = 64
hidden_dim = 128
num_characters = len(char_to_index)  # Based on the char_to_index mapping
num_tags = len(tag_to_index)         # Based on the tag_to_index mapping

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

# Initialize model
model = BiLSTMTagger(num_characters, num_tags, embedding_dim, hidden_dim)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Load data from JSON file
data = load_data(data_location)

# Training loop
def train_model(model, data, epochs=5, batch_size=1):
    model.train()
    dataloader = DataLoader(DateDataset(data), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_progress = 0
        for characters, tags in dataloader:
            characters, tags = characters.to(device), tags.to(device)
            # Forward pass
            outputs = model(characters)
            loss = criterion(outputs.view(-1, num_tags), tags.view(-1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_progress += 1
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f} ({epoch_progress/len(data)}%)')
    
    torch.save(model, 'model.model')

if __name__ == '__main__':
    train_model(model, data)