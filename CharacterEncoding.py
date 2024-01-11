import json
import string

def find_unique_characters_from_file(filename):
    # Read the JSON file
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Initialize a set to store unique characters
    unique_characters = set()

    # Iterate through the dataset and accumulate unique characters
    for date in data:
        for char, _ in date:
            unique_characters.add(char)

    return unique_characters

def find_unique_tags_from_file(filename):
    # Read the JSON file
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Initialize a set to store unique characters
    unique_tags = set()

    # Iterate through the dataset and accumulate unique characters
    for date in data:
        for _, tag in date:
            unique_tags.add(tag)

    return unique_tags

def encode_characters(char_list):
    index = 0
    encoded_chars = {}
    for char in char_list:
        encoded_chars[char] = index
        index += 1
    
    return encoded_chars

def generate_encoded_characters(filename):
    alphabet = list(string.ascii_lowercase)
    digits = list(string.digits)
    tokens = alphabet + digits + [' ']

    unique_chars = find_unique_characters_from_file(filename)

    # Find any non-alphanumeric characters that are included in given dataset
    for char in unique_chars:
        if char not in tokens:
            tokens.append(char)


    encoded_chars = encode_characters(tokens)

    return encoded_chars

if __name__ == '__main__':
    print(generate_encoded_characters('dataset.json'))
    print(sorted(find_unique_tags_from_file('dataset.json')))