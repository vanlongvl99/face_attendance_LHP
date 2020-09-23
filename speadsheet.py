import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint
from datetime import date
# import datetime
scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name("attendancelhp.json", scope)
# creds = ServiceAccountCredentials.from_json_keyfile_name("client_secret.json", scope)
# creds = ServiceAccountCredentials.from_json_keyfile_name("attendanceLHP-ca12644dc942.json", scope)


client = gspread.authorize(creds)

sheet = client.open("test_attendance").sheet1  # Open the spreadhseet

# print(sheet)
data = sheet.get_all_records()  # Get a list of all records
# print(data)
row = sheet.row_values(1)  # Get a specific row
col = sheet.col_values(2)  # Get a specific column
# print(type(row))
# time = str(date.today())
# print(time)
print("row 1",row)
print("col 2",col)
name_peron = "van_long"
index_col = col.index(name_peron) + 1
index_row = 7
# print(index)
sheet.update_cell(index_col, index_row,"có mặt")
# cell = sheet.cell(1,2).value  # Get the value of a specific cell
# 
# insertRow = ["hello", 5, "red", "blue"]
# sheet.add_rows(insertRow, 4)  # Insert the list as a row at index 4
# 
# sheet.update_cell(2,2, "CHANGED")  # Update one cell
# 
# numRows = sheet.row_count  # Get the number of rows in the sheet