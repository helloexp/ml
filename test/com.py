from test.util import DateUtils

s="/home/hdfs/cdp/work/tmp/cdp.sh %s > /home/hdfs/cdp/work/log/label_report_%s.log"

start="20161001"
end="20161101"

start_date = DateUtils.str2date(start)

end_date = DateUtils.str2date(end)


while not DateUtils.is_same_day(start_date,end_date):

    date_str = DateUtils.date2str(start_date)
    commond = s % (date_str,date_str)

    start_date=DateUtils.plus_day(start_date,1)

    print(commond)










