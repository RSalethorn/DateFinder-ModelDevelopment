import pandas
from DateModelList import DateModelList

list_location = "model_list.json"
out_location = "date_models.xlsx"

model_list = DateModelList(list_location)

json_data = model_list.readAll()

df = pandas.DataFrame(json_data)

df.to_excel(out_location)

