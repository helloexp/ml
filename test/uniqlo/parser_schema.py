# coding:utf-8
import json

import xlrd

merge_schema=[
    # {"dw_01_member":["dw_31_tmall_head","tmall_head"]},
    # {"dw_02_product_category":["dw_02_product","product"]},
    # {"product":["dw_tmall_detail","tmall_detail_product"]},
    # {"tmall_head":["tmall_detail_product","tmall_detail"]}
    # {"dw_member":["dw_tmall_o2o_head","member_tmall_o2o_head"]},
    # {"dw_store":["member_tmall_o2o_head","tmall_o2o_head"]},
    # {"product":["dw_tmall_o2o_detail","product_tmall_o2o_detail"]},
    # {"tmall_o2o_head":["product_tmall_o2o_detail","tmall_o2o_detail"]},

    # {"dw_store":["dw_epay_payment","store_epay_payment"]},
    # {"dw_member":["store_epay_payment","epay_payment"]},

    # {"product":["dw_15_pos_detail","product_pos_detail"]},
    # {"product_pos_detail":["dw_12_epay_payment","dw_15_pos_detail"]}

    {"dw_01_member":["dw_81_ux_hotline","ux_hotline"]}
]

excel_path = "/Users/tong/Desktop/hypers/uniqlo/02.DW数据结构设计_v1.25.xlsx"

excel = xlrd.open_workbook(excel_path)

sheets = excel.sheet_names()
print(sheets)

gen = False

def join_parquet_field(left_key, right_value):
    if (right_value != ""):
        return left_key.strip() + ":" + right_value.strip(), "STRING"
    else:
        return left_key.strip(), "STRING"


def generate_schema(sheet_name):
    sheet = excel.sheet_by_name(sheet_name)
    nrows = sheet.nrows
    ncols = sheet.ncols
    # print("nrows", nrows)
    # print("ncols", ncols)

    res = []

    last_key = ""

    for r in range(3, nrows):
        left_key = sheet.cell(r, 2)
        if (left_key.value != ""):
            last_key = left_key

        right_value = sheet.cell(r, 3)
        field_name = sheet.cell(r, 5)

        pareuqt_field, type = join_parquet_field(last_key.value, right_value.value)

        field_map = {
            "name": field_name.value,
            "fieldName": pareuqt_field,
            "type": type
        }

        res.append(field_map)

        if(left_key.value=="ETL_FILE_LINE_NO"):
            break

    return res


channel_name_2_schema={}

for sheet in sheets:

    schema_map = {"channel": sheet, "fileType": "PARQUERT"}

    if (sheet == "dw_01_member"):
        gen = True

    if (gen):
        schema = generate_schema(sheet)
        schema_map.update({"fields":schema})
        channel_name_2_schema.update({sheet:schema_map})


def merge_field_key(fields,prefix):
    for f in fields:

        get = f.get("fieldName")

        f.update({"fieldName": (prefix + "_" + get).upper()})

# {"channel": "dw_tmall_head", "fileType": "PARQUERT", "fields": [{"type": "STRING", "fieldName": "DW_TMALL_HEAD_ORDER_NO", "name": "\u8ba2\u5355\u53f7"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_ORDER_TYPE:KEY", "name": "\u8ba2\u5355\u7c7b\u578bKEY"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_ORDER_TYPE:VALUE", "name": "\u8ba2\u5355\u7c7b\u578b\u540d\u79f0"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_TB_ORDER_NO", "name": "\u5929\u732b\u8ba2\u5355\u53f7"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_ORDER_DATE", "name": "\u4e0b\u5355\u65e5\u671f"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_PAY_DATE", "name": "\u4ed8\u6b3e\u65e5\u671f"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_DELIVER_DATE", "name": "\u53d1\u8d27\u65e5\u671f"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_EXPRESS_NAME", "name": "\u5feb\u9012\u540d\u79f0"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_EXPRESS_NO", "name": "\u5feb\u9012\u5355\u53f7"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_TB_USER_NO", "name": "\u6dd8\u5b9d\u7528\u6237\u540d"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_ALIPAY_USER_ID", "name": "\u963f\u91cc\u652f\u4ed8\u7528\u6237ID"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_MEMBER_ID", "name": "\u4f1a\u5458ID"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_CONSIGNEE:USER_NAME", "name": "\u6536\u8d27\u4eba"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_CONSIGNEE:PROVINCE", "name": "\u7701"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_CONSIGNEE:CITY", "name": "\u5e02/\u53bf"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_CONSIGNEE:DISTRICT", "name": "\u533a/\u9547/"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_CONSIGNEE:ADDRESS", "name": "\u5b8c\u6574\u5730\u5740"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_CONSIGNEE:ZIP", "name": "\u90ae\u7f16"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_CONSIGNEE:PHONE", "name": "\u6536\u8d27\u4eba\u7535\u8bdd"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_CONSIGNEE:MOBILE", "name": "\u6536\u8d27\u4eba\u624b\u673a"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_GOODS_FEE", "name": ""}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_FREIGHT_FEE", "name": ""}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_TOTAL_AMT", "name": "\u8ba2\u5355\u603b\u4ef7"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_INBOUND_DATE", "name": ""}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_BIZ_DATE", "name": "\u4e1a\u52a1\u5904\u7406\u65f6\u95f4"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_ETL_DATE", "name": "ETL\u52a0\u5de5\u65f6\u95f4"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_ETL_FILE_PATH", "name": "\u76f8\u5bf9SFTP\uff0c\u6587\u4ef6\u5168\u8def\u5f84"}, {"type": "STRING", "fieldName": "DW_TMALL_HEAD_ETL_FILE_LINE_NO", "name": "\u76f8\u5bf9\u4e8e\u6e90CSV\u7684\u884c\u53f7"}, {"type": "STRING", "fieldName": "DW_MEMBER_MEMBER_ID", "name": "\u4f1a\u5458ID"}, {"type": "STRING", "fieldName": "DW_MEMBER_MOBILE_NO", "name": "\u624b\u673a\u53f7\u7801"}, {"type": "STRING", "fieldName": "DW_MEMBER_NAME", "name": "\u59d3\u540d"}, {"type": "STRING", "fieldName": "DW_MEMBER_BIRTHDAY", "name": "\u751f\u65e5"}, {"type": "STRING", "fieldName": "DW_MEMBER_SEX:KEY", "name": "\u6027\u522bKEY"}, {"type": "STRING", "fieldName": "DW_MEMBER_SEX:CODE", "name": "\u6027\u522bCD"}, {"type": "STRING", "fieldName": "DW_MEMBER_SEX:VALUE", "name": "\u6027\u522b\u540d\u79f0"}, {"type": "STRING", "fieldName": "DW_MEMBER_LOCATION:PROVINCE", "name": "\u7701"}, {"type": "STRING", "fieldName": "DW_MEMBER_LOCATION:CITY", "name": "\u5e02"}, {"type": "STRING", "fieldName": "DW_MEMBER_LOCATION:DISTRICT", "name": "\u533a"}, {"type": "STRING", "fieldName": "DW_MEMBER_LOCATION:ADDRESS", "name": "\u8be6\u7ec6\u5730\u5740"}, {"type": "STRING", "fieldName": "DW_MEMBER_CLIENT_KEY", "name": ""}, {"type": "STRING", "fieldName": "DW_MEMBER_DATA_SOURCE:CHANNEL_ID", "name": "\u4f1a\u5458\u6765\u6e90\u6e20\u9053ID"}, {"type": "STRING", "fieldName": "DW_MEMBER_DATA_SOURCE:CHANNEL_NAME", "name": "\u4f1a\u5458\u6765\u6e90\u6e20\u9053\u540d\u79f0"}, {"type": "STRING", "fieldName": "DW_MEMBER_DATA_SOURCE:CHANNEL_SOURCE", "name": "\u4f1a\u5458\u6765\u6e90\u6e20\u9053\u6e90"}, {"type": "STRING", "fieldName": "DW_MEMBER_WECHAT:UNION_ID", "name": "\u5fae\u4fe1UNION_ID"}, {"type": "STRING", "fieldName": "DW_MEMBER_WECHAT:NICK_NAME", "name": "\u5fae\u4fe1\u7528\u6237\u540d"}, {"type": "STRING", "fieldName": "DW_MEMBER_WECHAT:EPAY_USER_ID", "name": "EPAY\u7528\u6237ID"}, {"type": "STRING", "fieldName": "DW_MEMBER_WECHAT:MCARD_ID", "name": "\u5fae\u4fe1\u8054\u540d\u4f1a\u5458\u5361ID"}, {"type": "STRING", "fieldName": "DW_MEMBER_WECHAT:MCARD_CREATE_TIME", "name": "\u5fae\u4fe1\u8054\u540d\u4f1a\u5458\u5361\u521b\u5efa\u65e5\u65f6"}, {"type": "STRING", "fieldName": "DW_MEMBER_WECHAT:MCARD_CHANNEL_ID", "name": "\u5fae\u4fe1\u8054\u540d\u4f1a\u5458\u5361\u6765\u6e90\u6e20\u9053ID"}, {"type": "STRING", "fieldName": "DW_MEMBER_WECHAT:MCARD_CHANNEL_NAME", "name": "\u5fae\u4fe1\u8054\u540d\u4f1a\u5458\u5361\u6765\u6e90\u6e20\u9053\u540d\u79f0"}, {"type": "STRING", "fieldName": "DW_MEMBER_WECHAT:MCARD_CHANNEL_SOURCE", "name": "\u5fae\u4fe1\u8054\u540d\u4f1a\u5458\u5361\u6765\u6e90\u6e20\u9053\u6e90"}, {"type": "STRING", "fieldName": "DW_MEMBER_ALIPAY:USER_ID", "name": "\u652f\u4ed8\u5b9d\u7528\u6237ID"}, {"type": "STRING", "fieldName": "DW_MEMBER_ALIPAY:MCARD_ID", "name": "\u963f\u91cc\u8054\u540d\u4f1a\u5458\u5361ID"}, {"type": "STRING", "fieldName": "DW_MEMBER_ALIPAY:MCARD_CREATE_TIME", "name": "\u963f\u91cc\u8054\u540d\u4f1a\u5458\u5361\u521b\u5efa\u65e5\u65f6"}, {"type": "STRING", "fieldName": "DW_MEMBER_ALIPAY:MCARD_CHANNEL_ID", "name": "\u963f\u91cc\u8054\u540d\u4f1a\u5458\u5361\u6765\u6e90\u6e20\u9053ID"}, {"type": "STRING", "fieldName": "DW_MEMBER_ALIPAY:MCARD_CHANNEL_NAME", "name": "\u963f\u91cc\u8054\u540d\u4f1a\u5458\u5361\u6765\u6e90\u6e20\u9053\u540d\u79f0"}, {"type": "STRING", "fieldName": "DW_MEMBER_ALIPAY:MCARD_CHANNEL_SOURCE", "name": "\u963f\u91cc\u8054\u540d\u4f1a\u5458\u5361\u6765\u6e90\u6e20\u9053\u6e90"}, {"type": "STRING", "fieldName": "DW_MEMBER_TAOBAO:USER_ID", "name": "\u6dd8\u5b9d\u7528\u6237ID"}, {"type": "STRING", "fieldName": "DW_MEMBER_TAOBAO:NICK_NAME", "name": "\u6dd8\u5b9d\u7528\u6237\u540d"}, {"type": "STRING", "fieldName": "DW_MEMBER_QQ:USER_ID", "name": "QQ\u7528\u6237ID"}, {"type": "STRING", "fieldName": "DW_MEMBER_QQ:NICK_NAME", "name": "QQ\u7528\u6237\u540d"}, {"type": "STRING", "fieldName": "DW_MEMBER_CREATE_TIME", "name": "\u521b\u5efa\u65e5\u65f6"}, {"type": "STRING", "fieldName": "DW_MEMBER_UPDATE_TIME", "name": "\u66f4\u65b0\u65e5\u65f6"}, {"type": "STRING", "fieldName": "DW_MEMBER_BIZ_DATE", "name": "\u4e1a\u52a1\u5904\u7406\u65f6\u95f4"}, {"type": "STRING", "fieldName": "DW_MEMBER_ETL_DATE", "name": "ETL\u52a0\u5de5\u65f6\u95f4"}, {"type": "STRING", "fieldName": "DW_MEMBER_ETL_FILE_PATH", "name": "\u76f8\u5bf9SFTP\uff0c\u6587\u4ef6\u5168\u8def\u5f84"}, {"type": "STRING", "fieldName": "DW_MEMBER_ETL_FILE_LINE_NO", "name": "\u76f8\u5bf9\u4e8e\u6e90CSV\u7684\u884c\u53f7"}]}


for name_2_schema in channel_name_2_schema.items():

    fields = name_2_schema[1].get("fields")

    for f in fields:
        get = f.get("fieldName")

        f.update({"fieldName": (name_2_schema[0]+"_"+get).upper()})



for mergeItem in merge_schema:
    items = mergeItem.items()

    for item in items:

        need_merge_schema = channel_name_2_schema.pop(item[0])
        intoItem = item[1][0]
        into_schema = channel_name_2_schema.pop(intoItem)

        fields = need_merge_schema.get("fields")

        # if(intoItem in set(sheets)):
        #     merge_field_key(fields,item[0])

        into_fields = into_schema.get("fields")

        # merge_field_key(into_fields, intoItem)

        into_fields.extend(fields)

        new_channel_name = item[1][1]
        into_schema.update({"channel": new_channel_name})

        channel_name_2_schema.update({new_channel_name:into_schema})

        print(json.dumps(into_schema))
