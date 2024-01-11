import re

def tokenize_dates(date_strings):
        # Regex pattern: splits on punctuation and spaces, keeping them as separate tokens
        pattern = r'([ \W])'

        # Tokenize each date string
        tokenized_dates = [re.split(pattern, date) for date in date_strings]

        # Filter out empty strings that are not spaces
        tokenized_dates = [[token for token in date if token or token == ' '] for date in tokenized_dates]

        return tokenized_dates

def add_quotes(text):
    return f"'{text}'"

def get_ordinal_indicator(day):
    if 10 <= day % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    return suffix

# Takes a list stored as a string (i.e. "['3', 'we', 'jan']")
# and seperates into an actual python list
def string_list_seperator(string_list):
    format_list = string_list.strip('][').split(', ')
    for i in range(len(format_list)):
        format_list[i] = format_list[i].replace("'","")
        format_list[i] = format_list[i].replace("\\\\","\\") 

    return format_list