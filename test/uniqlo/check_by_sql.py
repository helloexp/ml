#coding:utf-8
import json
import os

import sys

#{"sqoop":"SELECT count(distinct f.ZFB_UID) FROM [bi_rpt].[dbo].[pj012_acpf] as f where cast(f.CREATE_TIMESTAMP as date) <= cast('20180627' as date)","hive":"select count(distinct(member.zfb_uid)) from mdm_ods_pre.ods_01_member as member where to_date(create_timestamp)<=\"2018-06-27\";"}
import traceback

BASE_SQOOP="""sqoop eval --connect "jdbc:sqlserver://172.16.3.61:1433;DatabaseName=crm_ds" --username sqoop_hypers --password Uniqlo@2018   --query \"%s\" """
HIVE="hive -e '%s'"

def query(sql):
    popen = os.popen(sql)
    res=popen.read()
    popen.close()
    return res

def deal_sqoop(sqoop_res):
    sqoop_split = sqoop_res.split("\n")
    return sqoop_split[-3].replace("|", "").strip()

def dealt_hive(sql_res):
    res = sql_res.split("\n")[0]
    return res.strip()

def do_query(sql):
    sqoop_sql = BASE_SQOOP % sql.get("sqoop")
    hive_sql = HIVE % sql.get("hive")

    sqoop_res = query(sqoop_sql)
    hive_res = query(hive_sql)

    print(sqoop_sql, sqoop_res)
    print(hive_sql,hive_res)

    sqoop_res=deal_sqoop(sqoop_res)
    hive_res = dealt_hive(hive_res)

    print(sqoop_res,hive_res)
    assert sqoop_res==hive_res


if __name__ == '__main__':

    args = sys.argv[1]

    sql_file = open(args,"r")

    sqls = sql_file.readlines()

    res=[]
    for sql in sqls:
        sql = json.loads(sql)
        try:
            do_query(sql)
        except Exception as e:
            print('traceback.print_exc:',traceback.print_exc())
            res.append(sql)
    print(res)











