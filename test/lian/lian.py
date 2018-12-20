# coding=utf-8
import json

import xlrd

basepath = "/Users/tong/Desktop/hypers/lianhelihua/"
excel_path = "%sDW表设计 20181213.xlsx" % basepath
excel = xlrd.open_workbook(excel_path)

sheets = excel.sheet_names()

# print(sheets)


def generate_chema(sheet_name):

    res=[]
    sheet = excel.sheet_by_name(sheet_name)
    nrows = sheet.nrows

    for r in range(1, nrows):
        left_key = sheet.cell(r, 0)
        comment=sheet.cell(r, 1)

        field_map = {
            "name": left_key.value,
            "fieldName": comment.value,
            "type": "STRING"
        }

        res.append(field_map)

    return res


if __name__ == '__main__':
    sheet = "F04.fact_social_behavior"

    schema_map = {"channel": "dw."+sheet, "fileType": "PARQUERT"}
    chema = generate_chema(sheet)
    schema_map.update({"fields":chema})

    file_write = open(basepath + sheet,"w")
    jsons = json.dumps(schema_map,ensure_ascii=False).encode().decode()
    print(jsons)
    file_write.write(jsons)
    file_write.close()



