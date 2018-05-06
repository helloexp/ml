import csv

import sys

csv.field_size_limit(sys.maxsize)

print(sys.maxsize)

csv_file = open("/Users/tong/Downloads/part-00000.csv")
reader = csv.reader(csv_file)


for i in reader:

    

    pass









