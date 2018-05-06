import json

s="""{"campaignStatus": 1,
      "adgroupName": "视频广告",
      "campaignId": 84,
      "adgroupId": 1556,
      "accountStatus": 1,
      "accountName": "PM_WEM_DMP_01",
      "adId": 25327,
      "campaignName": "demo推广计划",
      "adStatus": 1,
      "adName": "视频广告",
      "adgroupStatus": 1,
      "accountId": 33}"""
load = json.loads(s)

res=[]
for k in load.keys():
    if(k.find("Name")):
        res.append("private String "+k+";")
    else:
        res.append("private Long "+k+";")

for r in res:
    print(r)