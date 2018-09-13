
will_delete_tables="""hdmp:atomic_segment_report
hdmp:atomic_segment_report_ds
hdmp:atomic_segment_report_ds_test
hdmp:atomic_segment_report_test
hdmp:cookie_list_mma
hdmp:cookie_list_pc
hdmp:label_report_android
hdmp:label_report_android_test
hdmp:label_report_ds_android
hdmp:label_report_ds_ios
hdmp:label_report_ds_others
hdmp:label_report_ds_pc
hdmp:label_report_ds_union
hdmp:label_report_ios
hdmp:label_report_others
hdmp:label_report_pc
hdmp:label_result_ds
hdmp:label_result_ds_test
hdmp:label_result_new
hdmp:segment_report
hdmp:segment_report_ds
hdmp:ta_segment
hdmp:ta_segment_ds"""



delete="""hdmp:label_report_ds_android_test
hdmp:label_report_ds_ios_test
hdmp:label_report_ds_others_test
hdmp:label_report_ds_pc_test
hdmp:label_report_ds_union_test
hdmp:label_report_ios_test
hdmp:label_report_others_test
hdmp:label_report_pc_copy
hdmp:label_report_pc_test
hdmp:label_report_v3_test
hdmp:label_report_v4
hdmp:label_result
hdmp:label_result_test
hdmp:label_result_v3
hdmp:label_result_v3_test
hdmp:segment_import_v3
hdmp:segment_report_test_ds
hdmp:segment_report_test_tmp
hdmp:segment_report_v3
hdmp:segment_report_v3_test"""

s1 = set(will_delete_tables.split("\n"))
print(s1)

s2=set(delete.split("\n"))

print(s1.intersection(s2))

import os

os.system("hadoop fs -ls /hbase/data/")

