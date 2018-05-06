import csv




def read_csv(filename,inter=","):

    res=[]
    reader = csv.reader(open(filename, "r"))
    for line in reader:
        res.append(line)

    return res

def split_tab_content(lines):
    res={}
    for line in lines:
        line_split = line[0].split("\t")
        device_id=line_split[0]
        phone=line_split[1]
        res.update({phone:device_id})

    return res

def split_dmp_device(lines):
    res = {}
    for line in lines:
        device_id = line[0]
        phone = line[1]
        res.update({phone: device_id})

    return res

idfa_device_2_phone = read_csv("/Users/tong/Downloads/work/idmapping/DEVICE_CONVERSION_ACTION_6900,6903_idfa_20180204_20180204.csv")
idfa_device_2_phone=split_tab_content(idfa_device_2_phone)

imei_device_2_phone = read_csv("/Users/tong/Downloads/work/idmapping/DEVICE_CONVERSION_ACTION_6900,6903_imei_20180204_20180204.csv")
imei_device_2_phone=split_tab_content(imei_device_2_phone)

print(idfa_device_2_phone)
print(imei_device_2_phone)

dmp_device_2_phone = read_csv("/Users/tong/Downloads/work/idmapping/device_id-mobile.csv")
print(len(dmp_device_2_phone))



dmp_device_2_phone = split_dmp_device(dmp_device_2_phone)

print(("dmp",len(dmp_device_2_phone)))


phone_hfa_num = set(idfa_device_2_phone.keys()) | set(imei_device_2_phone.keys())
print(type(phone_hfa_num))
print(len(phone_hfa_num))








