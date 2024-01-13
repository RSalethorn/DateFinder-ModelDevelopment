import json
import os

class DateModelList:
    def __init__(self, list_location):
        self.list_location = list_location

        if os.path.exists(self.list_location) is False:
            file_data = []
            self.writeAll(file_data)

    def readAll(self):
        with open(self.list_location, "r+") as file:
            file_data = json.load(file)
        return file_data
    
    def writeAll(self, file_data):
        with open(self.list_location, 'w+') as file:
                file.write(json.dumps(file_data, indent=4))


    def addModel(self, model_data):
        file_data = self.readAll()

        if file_data == None:
            file_data = []

        file_data.append(model_data)

        self.writeAll(file_data)

    
    
    def changeValue(self, model_name, key, value):
        file_data = self.readAll()
        for i in range(len(file_data)):
            if file_data[i]["name"] == model_name:
                file_data[i][key] = value
                break

        self.writeAll(file_data)


if __name__ == "__main__":
    model_list = DateModelList("date_model_list.json")
    model_list.changeValue("Odie_Lemke", "test", 1)

