import json
import random
from Date import DateFormat
from DateHelper import string_list_seperator

class DatasetBuilder:
    def __init__(self, combinations_filename, dataset_filename, length):
        self.combinations_filename = combinations_filename
        self.dataset_filename = dataset_filename
        self.length = length

    def build(self):
        print(f"Dataset build started, combinations retrieved from '{self.combinations_filename}'")
        combinations_json = self.load_combinations_file(self.combinations_filename)
        tagged_dates = []
        for x in range(self.length):
            combination = random.choice(combinations_json)
            date = DateFormat(
                combination["format_string"],
                string_list_seperator(combination["format_list"]),
                combination["date_order"]
                )
            tagged_date = date.generate_random_datetime_tagged()
            tagged_dates.append(tagged_date)
            print(f"\rDate {x+1}/{self.length} completed", end='', flush=True)

        print(f"\nBeginning to save dataset to file, location is {self.dataset_filename}")
        self.save_dataset_file(self.dataset_filename, tagged_dates)
        print(f"Dataset build finished, saved to '{self.dataset_filename}'")

    def load_combinations_file(self, filename):
        with open(filename, "r") as file:
            data = json.load(file)

        return data
    
    def save_dataset_file(self, dataset_filename, dataset_list):
        with open(dataset_filename, 'w+') as file:
            file.write(json.dumps(dataset_list, indent=4))

if __name__ == "__main__":
    dsb = DatasetBuilder("combinations.json", "dataset.json", 100000)
    dsb.build()
