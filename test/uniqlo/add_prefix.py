# coding:utf-8
import json

import xlrd

excel_path_14 = "/Users/tong/Desktop/uniqlo/02.DW数据结构设计_v0.14.xlsx"
excel_path_19 = "/Users/tong/Desktop/uniqlo/02.DW数据结构设计_v0.19.xlsx"


def read_sheets(path):
    excel = xlrd.open_workbook(path)
    sheets = excel.sheet_names()
    sheets = list(filter(lambda x: x.startswith("dw"), sheets))
    return sheets

sheets_14 = read_sheets(excel_path_14)
sheets_19 = read_sheets(excel_path_19)

res={}


for sheet in sheets_19:
    tmp=[]
    split_sheets = sheet.split("_")
    tmp.append(split_sheets[0])
    tmp.extend(split_sheets[2:])
    key = "_".join(tmp)
    res.update({key:"dw_"+split_sheets[1]})

print(json.dumps(res))



print(set(res.keys()).difference(set(sheets_14)))

j = """{
	"dw_store": "dw_04",
	"dw_product_category": "dw_02_cate",
	"dw_epay_return": "dw_13",
	"dw_tmall_o2o_detail": "dw_34",
	"dw_tmall_head": "dw_31",
	"dw_epay_payment": "dw_12",
	"dw_pos_detail": "dw_15",
	"dw_member": "dw_01",
	"dw_tmall_detail": "dw_32",
	"dw_tmall_o2o_head": "dw_33",
	"dw_product": "dw_02"
}"""

schema_dict = json.loads(j)

for k,v in schema_dict.items():
    print(",".join([k,v]))
