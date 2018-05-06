
import json

f = open("./2a3836be-d7cb-4d90-99a6-6529a2b9aee5.json","r")

lines = f.readlines()

w = open("./1753.txt", "w")

for line in lines:

    content = json.loads(line)
    ads = content["ads"]

    for ad in ads:
        campaignId = ad["campaignId"]
        if campaignId ==1753:
            print(line)
            w.write(line)
            break