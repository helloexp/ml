import os

import hashlib
import csv

import sys

csv_file_path = "/Users/tong/IdeaProjects/work/uniqlo_identify/uniqlo/spark-warehouse/labeledId.csv"

m2 = hashlib.md5()


csv_file = open(csv_file_path, "w")
writer = csv.writer(csv_file)
writer.writerow(["session","uid","id","date"])
# CREATE TABLE IF NOT EXISTS labeled_id(session string, uid string, id string, date string) STORED AS carbondata
# session string, uid string, id string, date string
lines = 10000000

div_people=1000

for i in range(0, lines):
    m2.update(str(i%div_people).encode())
    md5 = m2.hexdigest()

    tmp=[md5,i%10,i%100+1000,30-i%30]
    writer.writerow(tmp)


csv_file.close()






