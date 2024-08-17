import csv
with open('C:\\Users\\Gowtham\\Documents\\sample1.csv') as sam:
  file=csv.reader(sam)
  for row in file:
    print(row)
