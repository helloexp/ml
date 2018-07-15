#coding:utf-8
import os
import zipfile

import xlrd

excel_path = "/Users/tong/Desktop/uniqlo/azkaban.xlsx"


excel = xlrd.open_workbook(excel_path)

content="""type=command
command=${base_path}/${%s}/%s
dependencies=%s
"""
# flow.name: tail job name
subflow_content_with_dependencies="""type=flow
flow.name=%s
dependencies=%s
"""

subflow_content_no_dependencies="""type=flow
flow.name=%s
"""

sheets = excel.sheet_names()

id2name={"1":"ods","2":"dw","3":"insight","4":"dm"}



for sheet in sheets:
    sheet = excel.sheet_by_name(sheet)
    nrows = sheet.nrows
    ncols = sheet.ncols

    for r in range(1, nrows):

        full_name = sheet.cell(r,3).value
        depend = sheet.cell(r,4).value.replace(".sh", "")
        tail= sheet.cell(r,5).value

        file_job = open("./job/" + full_name.replace(".sh", ".job"),"w")

        modle_order = full_name[0]
        shell_dir = id2name.get(modle_order)

        if("subflow" in full_name):
            if(depend!=""):
                file_job.write(subflow_content_with_dependencies % (tail,depend))
            else:
                file_job.write(subflow_content_no_dependencies % (tail))
        else:
            file_job.write(content % (shell_dir,full_name,depend))
        file_job.close()




def write_property():
    properties = open("./job/uniqlo.properties","w")
    properties.write("base_path=/home/hdfs/hypers\n")
    for k,v in id2name.items():
        properties.write(v+"="+"0"+k+"_"+v+"\n")
    properties.close()

write_property()




def write_zip():
    path = "./uniqlo.zip"
    zpfd = zipfile.ZipFile(path, mode='w')

    job_path = "./job/"
    listdir = os.listdir(job_path)

    for f in listdir:
        zpfd.write(job_path+f,f)

    zpfd.close()

write_zip()
