import torch
import torch
import torch.nn as nn
from datetime import datetime
import math
import json
import random
from Date import DateFormat

def preprocess_input(input_date, char_to_index):
    input_date = input_date.lower()
    # Tokenize and convert to indices
    character_indices = [char_to_index[char] for char in input_date]
    # Convert to tensor and add batch dimension
    return torch.tensor(character_indices, dtype=torch.long).unsqueeze(0).to(device)

def postprocess_output(output_tensor, index_to_tag):
    # Convert probabilities to tag indices
    _, predicted_tags = torch.max(output_tensor, dim=2)

    if predicted_tags.is_cuda:
        predicted_tags = predicted_tags.cpu()

    # Convert indices to tags
    predicted_tags = [index_to_tag[index.item()] for index in predicted_tags[0]]
    return predicted_tags

def predict_date_tags(model, input_date, char_to_index, index_to_tag):
    # Preprocess the input
    input_tensor = preprocess_input(input_date, char_to_index)

    # Set the model to evaluation mode
    model.eval()

    # Predict
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Postprocess the output
    predicted_tags = postprocess_output(output_tensor, index_to_tag)

    return predicted_tags

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

def pretty_print_tags(tag_list):
    chars = []
    tags = [[], [], []]
    for char, tag in tag_list:
        chars.append(char)
        tag = list(tag)
        for x in range(len(tags)):
            if tag[x] == "N":
                tag[x] = " "
            tags[x].append(tag[x])
    original_date = "".join(chars)
    table_width = (len(original_date) * 2) + 1

    border_difference = table_width - len(original_date)
    border = "=" * math.floor((border_difference / 2))
    
    header = f"{border} {original_date} {border}"[:table_width]

    print(header)
    print('|' + '|'.join(chars) + '|')
    print('|' + '|'.join(tags[0]) + '|')
    print('|' + '|'.join(tags[2]) + '|')
    print(f"{'=' * table_width}")


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
index_to_tag = {
                0: 'O-None',
                1: 'B-B', 
                2: 'I-B',
                3: 'B-H', 
                4: 'I-H',
                5: 'B-I', 
                6: 'I-I',
                7: 'B-M', 
                8: 'I-M', 
                9: 'B-S', 
                10: 'I-S', 
                11: 'B-Y', 
                12: 'I-Y', 
                13: 'B-Z', 
                14: 'I-Z', 
                15: 'B-b', 
                16: 'I-b', 
                17: 'B-d', 
                18: 'I-d',
                19: 'B-f', 
                20: 'I-f',
                21: 'B-m', 
                22: 'I-m',
                23: 'B-p', 
                24: 'I-p', 
                25: 'B-y', 
                26: 'I-y',
                27: 'B-z', 
                28: 'I-z'     
            }



# Load the saved model weights
model = torch.load('model_Gloria_Osinski.model')
# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

with open('combinations.json', 'r') as file:
    combinations = json.load(file)
total_tags = 0
correct_tags = 0
perfect_date_count = 0

total_dates = 10000
for x in range(total_dates):
    combo = random.choice(combinations)
    format_list = combo["format_list"].strip('][').split(', ')
    for i in range(len(format_list)):
        format_list[i] = format_list[i].replace("'","")
        format_list[i] = format_list[i].replace("\\\\","\\")
    date_format = DateFormat(combo["format_string"], format_list, combo["date_order"])
    date_tagged = date_format.generate_random_datetime_tagged()
    date = ""
    for char in date_tagged:
        date += char[0]
    predicted_tags = predict_date_tags(model, date, char_to_index, index_to_tag)
    chars_tags = list(zip(date, predicted_tags))
    
    perfect_date = True
    for x in range(len(chars_tags)):
        total_tags += 1
        if chars_tags[x][1] == date_tagged[x][1]:
            correct_tags += 1
        else:
            print(f"Fail on: {date}")
            perfect_date = False
    if perfect_date == True:
        perfect_date_count += 1
percentage_tags = correct_tags/total_tags * 100
percentage_dates = perfect_date_count/total_dates * 100
print(f"Correct Tags: {correct_tags}/{total_tags} ({percentage_tags}%)")
print(f"Perfect Dates: {perfect_date_count}/{total_dates} ({percentage_dates}%)")

running = True

while running == True:
    input_date = input("Enter a date (Enter quit to stop): ")
    if input_date == "quit":
        break
    before = datetime.now()
    predicted_tags = predict_date_tags(model, input_date, char_to_index, index_to_tag)
    after = datetime.now()
    total_time = after - before
    chars_tags = list(zip(input_date, predicted_tags))
    pretty_print_tags(chars_tags)
    print(f"Time Taken: {total_time}")