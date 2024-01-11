from datetime import datetime, timedelta
import random
import pytz

from DateArrayPlacement import DateArrayPlacement
from DateHelper import get_ordinal_indicator

class DateFormat:
    def __init__(self, format_string, format_list=None, date_order=None):
        self.format_string = format_string
        self.format_list = format_list
        self.date_order = date_order

        self.dap = DateArrayPlacement()

    def generate_random_datetime_string(self):
        random_datetime = self.get_random_time()

        # Format the random datetime using the chosen date and time format
        formatted_datetime = random_datetime.strftime(self.format_string)

        formatted_datetime = formatted_datetime.replace("$o", get_ordinal_indicator(random_datetime.day))

        return formatted_datetime
    
    def generate_random_datetime_tagged(self):
        if self.format_list == None or self.date_order == None:
            raise ValueError("DateFormat objects need format_list and date_order to generate tagged datetimes.")

        random_datetime = self.get_random_time()

        tagged_characters = []
        formatted_date_list = [""] * len(self.format_list)
        # Loops through the different parts that make up format
        # Array that is described on google sheets
        for i in range(len(self.format_list)):
            # Turn format part into its string equivalent
            formatted_date_list[i] = random_datetime.strftime(self.format_list[i])
            if '$o' in formatted_date_list[i]:
                formatted_date_list[i] = formatted_date_list[i].replace("$o", get_ordinal_indicator(random_datetime.day))
            
            if "%" in self.format_list[i]:
                token_class = self.format_list[i].replace("%", "")
            else:
                token_class = None


            #Split format part string into characters
            tokens = list(formatted_date_list[i])
            
            first_token = True
            for token in tokens:
                tag = "O"   
                if token_class != None and first_token == True:
                    first_token = False
                    tag = "B"
                elif token_class != None and first_token == False:
                    tag = "I"
                token = token.lower()
                tagged_characters.append([token, f"{tag}-{token_class}"])
        return tagged_characters
        
    def get_random_time(self):
        current_datetime = datetime.now()
        
        # Generate a random timedelta within a range
        random_days = timedelta(days=random.randint(1, 30000))
        
        random_seconds = timedelta(seconds=random.randint(1, 86400)) # Amount of seconds in 1 day
        
        random_microseconds = timedelta(microseconds=random.randint(1, 1000000))

        random_timezone = random.choice(pytz.all_timezones)
        timezone = pytz.timezone(random_timezone)

        current_datetime = timezone.localize(current_datetime)

        # Subtract the timedelta from the current datetime to get a random past datetime
        random_datetime = current_datetime - random_days - random_seconds - random_microseconds

        return random_datetime
    
    def generate_random_datetime_tagged_dmy(self):
        if self.format_list == None or self.date_order == None:
            raise ValueError("DateFormat objects need format_list and date_order to generate tagged datetimes.")

        random_datetime = self.get_random_time()

        if self.date_order == "m/d/y":
            dmy_1_class = "m"
            dmy_2_class = "d"
            dmy_3_class = "y"
        elif self.date_order == "d/m/y":
            dmy_1_class = "d"
            dmy_2_class = "m"
            dmy_3_class = "y"
        elif self.date_order == "y/m/d":
            dmy_1_class = "y"
            dmy_2_class = "m"
            dmy_3_class = "d"

        tagged_characters = []
        formatted_date_list = [""] * len(self.format_list)
        # Loops through the different parts that make up format
        # Array that is described on google sheets
        for i in range(len(self.format_list)):
            # Turn format part into its string equivalent
            formatted_date_list[i] = random_datetime.strftime(self.format_list[i])
            if '$o' in formatted_date_list[i]:
                formatted_date_list[i] = formatted_date_list[i].replace("$o", get_ordinal_indicator(random_datetime.day))
            
            if "%" in self.format_list[i]:
                token_class = self.format_list[i].replace("%", "")
            else:
                token_class = None


            #Split format part string into characters
            tokens = list(formatted_date_list[i])
            
            first_token = True
            for token in tokens:
                tag = "O"
                token_class = None
                if i == self.dap.dmy_1:
                    token_class = dmy_1_class
                elif i == self.dap.dmy_2:
                    token_class = dmy_2_class
                elif i == self.dap.dmy_3:
                    token_class = dmy_3_class
                    
                if token_class != None and first_token == True:
                    first_token = False
                    tag = "B"
                elif token_class != None and first_token == False:
                    tag = "I"
                token = token.lower()
                tagged_characters.append([token, f"{tag}-{token_class}"])
        return tagged_characters

