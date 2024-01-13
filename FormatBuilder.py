from itertools import product
import json
from datetime import datetime, timedelta
import random
import csv
import pytz
import re
import os

from datefinder.data_types.DateArrayPlacement import DateArrayPlacement
from datefinder.data_types.Date import DateFormat
from datefinder.helpers.DateHelper import add_quotes

class FormatBuilder:
    def __init__(self):
        self.dap = DateArrayPlacement()
        self.next_format_index = 0
        self.init_format_combinations()
        self.generate_formats_file("./data/formats/new_formats.json")
        
        
    def init_format_combinations(self):
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

        self.format_combinations = list(product(
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
        
        num_formats_before = len(self.format_combinations)
        
        self.format_combinations = self.remove_bad_format_combinations(self.format_combinations)

        num_formats_after = len(self.format_combinations)

        print(f"There are {num_formats_after} possible combinations of dates ({num_formats_before} before restrictions)")

    def generate_formats_file(self, filename):
        formats_file_data = []
        for format_combination in self.format_combinations:
            combination_data = self.generate_combination_data(format_combination)
            formats_file_data.append(combination_data)

        with open(filename, "w+", newline='') as file:
            file.write(json.dumps(formats_file_data, indent=4))

        print(f"Succesfully written combinations to '{filename}'")
        

    def remove_bad_format_combinations(self, format_combinations):
        # Restrictions
        format_combinations = self.remove_invalid_combination(format_combinations, "FirstComma", "Month Zero-padded")
        format_combinations = self.remove_invalid_combination(format_combinations, "SecondComma", "Month Zero-padded")

        format_combinations = self.remove_invalid_combination(format_combinations, "12hr (AM/PM)", "Timezone ISO8601 Format (Z is UTC)")

        format_combinations = self.remove_invalid_combination(format_combinations, "Time Not Included", "12hr (AM/PM)")
        format_combinations = self.remove_invalid_combination(format_combinations, "Time Not Included", "24hr")

        format_combinations = self.remove_invalid_combination(format_combinations, "Clock Seconds Not Included", "Clock Microseconds Included")

        format_combinations = self.remove_invalid_combination(format_combinations, "Clock Timezone Not Included", "Timezone With Text")
        format_combinations = self.remove_invalid_combination(format_combinations, "Clock Timezone Not Included", "Timezone With Numbers")
        format_combinations = self.remove_invalid_combination(format_combinations, "Clock Timezone Not Included", "Timezone UTC Offset")
        format_combinations = self.remove_invalid_combination(format_combinations, "Clock Timezone Not Included", "Timezone ISO8601 Format (Z is UTC)")

        format_combinations = self.remove_invalid_combination(format_combinations, "Time Not Included", "Clock Timezone Included")

        return format_combinations
    
    def generate_combination_data(self, format_combination):
        format_list = [""] * 84
        day_rep = ""
        month_rep = ""
        year_rep = ""
        if "/" in format_combination:
            format_list[self.dap.seperator_1] = "/"
            format_list[self.dap.seperator_2] = "/"
        if "\\" in format_combination:
            format_list[self.dap.seperator_1] = "\\"
            format_list[self.dap.seperator_2] = "\\"
        if "-" in format_combination:
            format_list[self.dap.seperator_1] = "-"
            format_list[self.dap.seperator_2] = "-"
        if "." in format_combination:
            format_list[self.dap.seperator_1] = "."
            format_list[self.dap.seperator_2] = "."
        if "," in format_combination:
            format_list[self.dap.seperator_1] = ","
            format_list[self.dap.seperator_2] = ","
        if "Space" in format_combination:
            format_list[self.dap.seperator_1] = " "
            format_list[self.dap.seperator_2] = " "
        if "FirstComma" in format_combination:
            format_list[self.dap.seperator_1] = ","
            format_list[self.dap.seperator_1_space_a] = " "
            format_list[self.dap.seperator_2] = " "
        if "SecondComma" in format_combination:
            format_list[self.dap.seperator_1] = " "
            format_list[self.dap.seperator_2] = ","
            format_list[self.dap.seperator_2_space_a] = " "

        if "Day non-zero padded" in format_combination: # TODO: WASN'T WORKING
            day_rep = "%d"
        if "Day zero padded" in format_combination:
            day_rep = "%d"
        if "Day ordinal" in format_combination:
            day_rep = "%d"

        if "Month non-zero padded" in format_combination: # TODO: WASN'T WORKING
            month_rep = "%m"
        if "Month Zero-padded" in format_combination:
            month_rep = "%m"
        if "Month As Text" in format_combination:
            month_rep = "%B"
        if "Month As Abrieviated Text" in format_combination:
            month_rep = "%b"

        if "Full Year" in format_combination:
            year_rep = "%Y"
        if "Shortened Year" in format_combination:
            year_rep = "%y"

        if "12hr (AM/PM)" in format_combination:
            format_list[self.dap.clock_h_space_b] = " "
            format_list[self.dap.clock_h] = "%I"
            format_list[self.dap.clock_sep_1] = ":"
            format_list[self.dap.clock_m] = "%M"
            format_list[self.dap.ampm] = "%p"
        if "24hr" in format_combination:
            format_list[self.dap.clock_h_space_b] = " "
            format_list[self.dap.clock_h] = "%H"
            format_list[self.dap.clock_sep_1] = ":"
            format_list[self.dap.clock_m] = "%M"
        
        if "Clock Seconds Included" in format_combination:
            format_list[self.dap.clock_sep_2] = ":"
            format_list[self.dap.clock_s] = "%S"
        
        if "Clock Microseconds Included" in format_combination:
            format_list[self.dap.clock_sep_3] = ":"
            format_list[self.dap.clock_ms] = "%f"

        if "Timezone With Text" in format_combination:
            format_list[self.dap.tz_label] = "%Z"
            format_list[self.dap.tz_label_space_b] = " "
        if "Timezone With Numbers" in format_combination:
            format_list[self.dap.tz_val] = "%z"
        if "Timezone UTC Offset" in format_combination:
            format_list[self.dap.tz_val_space_b] = " "
            format_list[self.dap.tz_val] = "UTC"
            format_list[self.dap.tz_label] = "%z"
        if "Timezone ISO8601 Format (Z is UTC)" in format_combination:
            format_list[self.dap.tz_val] = "Z"
            format_list[self.dap.tz_label] = "%z"

        if "Clock Alternate Position" in format_combination:
            for x in range(len(self.dap.default_clock_parts)):
                format_list[self.dap.alt_clock_parts[x]] = format_list[self.dap.default_clock_parts[x]]
                format_list[self.dap.default_clock_parts[x]] = ""

            format_list[self.dap.alt_tz_label_space_a] = " "
            format_list[self.dap.alt_clock_h_space_b] = ""
        
        if "m/d/y" in format_combination:
            format_list[self.dap.dmy_1] = month_rep
            format_list[self.dap.dmy_2] = day_rep
            format_list[self.dap.dmy_3] = year_rep

            if "Day ordinal" in format_combination:
                format_list[self.dap.dmy_2_ordinal] = "$o"

            date_order = "m/d/y"
        if "d/m/y" in format_combination:
            format_list[self.dap.dmy_1] = day_rep
            format_list[self.dap.dmy_2] = month_rep
            format_list[self.dap.dmy_3] = year_rep

            if "Day ordinal" in format_combination:
                format_list[self.dap.dmy_1_ordinal] = "$o"
            
            date_order = "d/m/y"
        if "y/m/d" in format_combination:
            format_list[self.dap.dmy_1] = year_rep
            format_list[self.dap.dmy_2] = month_rep
            format_list[self.dap.dmy_3] = day_rep

            if "Day ordinal" in format_combination:
                format_list[self.dap.dmy_3_ordinal] = "$o"
            
            date_order = "y/m/d"

        format_string = ''.join(format_list)

        example_datetimes = []

        date_format = DateFormat(format_string)
        for _ in range(3):
            example_datetimes.append(date_format.generate_random_datetime_string())

        combo_json = {
            "id": self.next_format_index,
            "format_string": format_string,
            "date_order": date_order,
            "format_list":str(format_list),
            "examples": example_datetimes
        }
        
        self.next_format_index += 1

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
    cb = FormatBuilder()

        