from itertools import product
import json
from datetime import datetime, timedelta
import random
import csv
import pytz
import re

from DateArrayPlacement import DateArrayPlacement
from Date import DateFormat
from DateHelper import add_quotes

class CombinationBuilder:
    def __init__(self):
        self.dap = DateArrayPlacement()
        self.next_combination_index = 0

        self.init_raw_combinations()
        self.generate_combinations_file("combinations.json")
        
        
    def init_raw_combinations(self):
        #TODO: Add non-zero padded hours, minutes? and seconds on clock
        orders = ["m/d/y", "d/m/y", "y/m/d"]
        date_seperators = ["/", "\\", "-", ".", ",", "Space", "FirstComma", "SecondComma"]#, " - (Spaces either side)", " . (Spaces either side)", " , (Spaces either side)", " / (Spaces either side)", " \\ (Spaces either side)"
        day_alternatives = ["Day zero padded", "Day ordinal"]#"Day non-zero padded", 
        month_alternatives = ["Month Zero-padded", "Month As Text", "Month As Abrieviated Text"]#"Month non-zero padded", 
        year_alternatives = ["Full Year", "Shortened Year"]
        time_included = ["Time Included", "Time Not Included"]
        clock_position = ["Clock Default Position", "Clock Alternate Position"]
        clock_type = ["12hr (AM/PM)", "24hr"]
        seconds_included = ["Clock Seconds Included", "Clock Seconds Not Included"]
        milliseconds_included = ["Clock Microseconds Included", "Clock Microseconds Not Included"]
        timezone_included = ["Clock Timezone Included", "Clock Timezone Not Included"]
        timezone_alternatives = ["Timezone With Text", "Timezone With Numbers", "Timezone UTC Offset", "Timezone ISO8601 Format (Z is UTC)"]

        self.raw_combinations = list(product(
            orders, 
            date_seperators, 
            day_alternatives, 
            month_alternatives, 
            year_alternatives,
            time_included,
            clock_position,
            clock_type,
            seconds_included,
            milliseconds_included,
            timezone_included,
            timezone_alternatives
        ))
        
        no_combinations_before = len(self.raw_combinations)
        
        self.raw_combinations = self.remove_bad_combinations(self.raw_combinations)

        no_combinations_after = len(self.raw_combinations)

        print(f"There are {no_combinations_after} possible combinations of dates ({no_combinations_before} before restrictions)")

    def generate_combinations_file(self, filename):
        combination_json = []
        for raw_combination in self.raw_combinations:
            combination_data = self.generate_combination_data(raw_combination)
            combination_json.append(combination_data)

        with open(filename, "w+", newline='') as file:
            file.write(json.dumps(combination_json, indent=4))

        print(f"Succesfully written combinations to '{filename}'")
        

    def remove_bad_combinations(self, raw_combinations):
        # Restrictions
        raw_combinations = self.remove_invalid_combination(raw_combinations, "FirstComma", "Month Zero-padded")
        raw_combinations = self.remove_invalid_combination(raw_combinations, "SecondComma", "Month Zero-padded")

        raw_combinations = self.remove_invalid_combination(raw_combinations, "12hr (AM/PM)", "Timezone ISO8601 Format (Z is UTC)")

        raw_combinations = self.remove_invalid_combination(raw_combinations, "Time Not Included", "12hr (AM/PM)")
        raw_combinations = self.remove_invalid_combination(raw_combinations, "Time Not Included", "24hr")

        raw_combinations = self.remove_invalid_combination(raw_combinations, "Clock Seconds Not Included", "Clock Microseconds Included")

        raw_combinations = self.remove_invalid_combination(raw_combinations, "Clock Timezone Not Included", "Timezone With Text")
        raw_combinations = self.remove_invalid_combination(raw_combinations, "Clock Timezone Not Included", "Timezone With Numbers")
        raw_combinations = self.remove_invalid_combination(raw_combinations, "Clock Timezone Not Included", "Timezone UTC Offset")
        raw_combinations = self.remove_invalid_combination(raw_combinations, "Clock Timezone Not Included", "Timezone ISO8601 Format (Z is UTC)")

        raw_combinations = self.remove_invalid_combination(raw_combinations, "Time Not Included", "Clock Timezone Included")

        return raw_combinations
    
    def generate_combination_data(self, raw_combination):
        format_list = [""] * 84
        day_rep = ""
        month_rep = ""
        year_rep = ""
        if "/" in raw_combination:
            format_list[self.dap.seperator_1] = "/"
            format_list[self.dap.seperator_2] = "/"
        if "\\" in raw_combination:
            format_list[self.dap.seperator_1] = "\\"
            format_list[self.dap.seperator_2] = "\\"
        if "-" in raw_combination:
            format_list[self.dap.seperator_1] = "-"
            format_list[self.dap.seperator_2] = "-"
        if "." in raw_combination:
            format_list[self.dap.seperator_1] = "."
            format_list[self.dap.seperator_2] = "."
        if "," in raw_combination:
            format_list[self.dap.seperator_1] = ","
            format_list[self.dap.seperator_2] = ","
        if "Space" in raw_combination:
            format_list[self.dap.seperator_1] = " "
            format_list[self.dap.seperator_2] = " "
        if "FirstComma" in raw_combination:
            format_list[self.dap.seperator_1] = ","
            format_list[self.dap.seperator_1_space_a] = " "
            format_list[self.dap.seperator_2] = " "
        if "SecondComma" in raw_combination:
            format_list[self.dap.seperator_1] = " "
            format_list[self.dap.seperator_2] = ","
            format_list[self.dap.seperator_2_space_a] = " "

        if "Day non-zero padded" in raw_combination: # TODO: WASN'T WORKING
            day_rep = "%d"
        if "Day zero padded" in raw_combination:
            day_rep = "%d"
        if "Day ordinal" in raw_combination:
            day_rep = "%d"

        if "Month non-zero padded" in raw_combination: # TODO: WASN'T WORKING
            month_rep = "%m"
        if "Month Zero-padded" in raw_combination:
            month_rep = "%m"
        if "Month As Text" in raw_combination:
            month_rep = "%B"
        if "Month As Abrieviated Text" in raw_combination:
            month_rep = "%b"

        if "Full Year" in raw_combination:
            year_rep = "%Y"
        if "Shortened Year" in raw_combination:
            year_rep = "%y"

        if "12hr (AM/PM)" in raw_combination:
            format_list[self.dap.clock_h_space_b] = " "
            format_list[self.dap.clock_h] = "%I"
            format_list[self.dap.clock_sep_1] = ":"
            format_list[self.dap.clock_m] = "%M"
            format_list[self.dap.ampm] = "%p"
        if "24hr" in raw_combination:
            format_list[self.dap.clock_h_space_b] = " "
            format_list[self.dap.clock_h] = "%H"
            format_list[self.dap.clock_sep_1] = ":"
            format_list[self.dap.clock_m] = "%M"
        
        if "Clock Seconds Included" in raw_combination:
            format_list[self.dap.clock_sep_2] = ":"
            format_list[self.dap.clock_s] = "%S"
        
        if "Clock Microseconds Included" in raw_combination:
            format_list[self.dap.clock_sep_3] = ":"
            format_list[self.dap.clock_ms] = "%f"

        if "Timezone With Text" in raw_combination:
            format_list[self.dap.tz_label] = "%Z"
            format_list[self.dap.tz_label_space_b] = " "
        if "Timezone With Numbers" in raw_combination:
            format_list[self.dap.tz_val] = "%z"
        if "Timezone UTC Offset" in raw_combination:
            format_list[self.dap.tz_val_space_b] = " "
            format_list[self.dap.tz_val] = "UTC"
            format_list[self.dap.tz_label] = "%z"
        if "Timezone ISO8601 Format (Z is UTC)" in raw_combination:
            format_list[self.dap.tz_val] = "Z"
            format_list[self.dap.tz_label] = "%z"

        if "Clock Alternate Position" in raw_combination:
            for x in range(len(self.dap.default_clock_parts)):
                format_list[self.dap.alt_clock_parts[x]] = format_list[self.dap.default_clock_parts[x]]
                format_list[self.dap.default_clock_parts[x]] = ""

            format_list[self.dap.alt_tz_label_space_a] = " "
            format_list[self.dap.alt_clock_h_space_b] = ""
        
        if "m/d/y" in raw_combination:
            format_list[self.dap.dmy_1] = month_rep
            format_list[self.dap.dmy_2] = day_rep
            format_list[self.dap.dmy_3] = year_rep

            if "Day ordinal" in raw_combination:
                format_list[self.dap.dmy_2_ordinal] = "$o"

            date_order = "m/d/y"
        if "d/m/y" in raw_combination:
            format_list[self.dap.dmy_1] = day_rep
            format_list[self.dap.dmy_2] = month_rep
            format_list[self.dap.dmy_3] = year_rep

            if "Day ordinal" in raw_combination:
                format_list[self.dap.dmy_1_ordinal] = "$o"
            
            date_order = "d/m/y"
        if "y/m/d" in raw_combination:
            format_list[self.dap.dmy_1] = year_rep
            format_list[self.dap.dmy_2] = month_rep
            format_list[self.dap.dmy_3] = day_rep

            if "Day ordinal" in raw_combination:
                format_list[self.dap.dmy_3_ordinal] = "$o"
            
            date_order = "y/m/d"

        format_string = ''.join(format_list)

        example_datetimes = []

        date_format = DateFormat(format_string)
        for _ in range(3):
            example_datetimes.append(date_format.generate_random_datetime_string())

        combo_json = {
            "id": self.next_combination_index,
            "format_string": format_string,
            "date_order": date_order,
            "format_list":str(format_list),
            "examples": example_datetimes
        }
        
        self.next_combination_index += 1

        return combo_json

    def remove_invalid_combination(self, list_of_lists, item1, item2):
        """
        Remove lists from a list of lists if they contain both item1 and item2.

        :param list_of_lists: List of lists to be filtered.
        :param item1: First item to check for.
        :param item2: Second item to check for.
        :return: Filtered list of lists.
        """
        return [lst for lst in list_of_lists if not (item1 in lst and item2 in lst)]

    
    

    
if __name__ == "__main__":
    cb = CombinationBuilder()

        