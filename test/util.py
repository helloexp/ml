# -*- coding: UTF-8 -*-
import calendar
import logging
import os
import threading
import traceback
from io import StringIO, BytesIO
import datetime
import boto
import boto.s3.connection
from boto.s3.key import Key
import math
import time
import sys
import platform
from ftplib import FTP

FTP_HOST = "10.10.10.62"
FTP_USER = "mtarget"
FTP_PASS = "irs123"

S3_HOST = "rest.irs01.com"
S3_ACCESS_KEY = "PB8F9DIIYO282BW0JZKH"
S3_SECRET_KEY = "tetM0D9lKHfmzriaqcbtIuV4GBNejWsEo1xBM0oA"
S3_BUCKET = ["routerdata1", "routerdata2"]
# S3_BUCKET=["test"]

per_thread_key_num = 200

DESTNITION_DIR = "f://opt/dar/" if platform.system() == "Windows" else "/data/router/"


def get_file_path(name, date_str):
    path = get_download_dir() + name + "/"
    file_path = path + date_str
    if not os.path.exists(path):
        os.mkdir(path)
    return file_path


def get_download_dir():
    return DESTNITION_DIR


class S3(object):
    def __init__(self, access_key, secret_key, hosts, bucket=None, bucket_key=None, file=None):
        self.access_key = access_key
        self.secret_key = secret_key
        self.hosts = hosts
        self.bucket_str = bucket
        self.key = bucket_key
        self.file = file
        self.conn = None
        self.bucket = None

    def __connect(self):  # _s3__connect

        self.conn = boto.connect_s3(
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                host=self.hosts,
                is_secure=True,
                calling_format=boto.s3.connection.OrdinaryCallingFormat(),
        )
        return self.conn

    def __get_bucket(self, bucket_str):
        if not self.conn:
            self.__connect()
        return self.conn.get_bucket(bucket_str)

    def list_bucket_files(self, bucket_str):
        bucket_keys = []
        bucket = self.__get_bucket(bucket_str)
        bucket_list_res = bucket.list()
        for blr in bucket_list_res:
            key_name = blr.name
            if key_name.endswith("tar.gz"):
                bucket_keys.append(key_name)
        return bucket_keys

    def __get_key(self, bucket, bucket_key):
        if not self.bucket:
            self.bucket = self.__get_bucket(bucket)
        return self.bucket.lookup(bucket_key)

    def get_file_to_disk(self, bucket, bucket_key, file_path):
        # file = open(file_path, "w")
        # self.__get_key(bucket_key).get_contents_to_file(file)
        # file.close()
        key = self.__get_key(bucket, bucket_key)
        key.get_contents_to_filename(file_path)
        return True

    def set_acl(self, bucket_key, type):
        """
        type:
            public-read
            private
        """
        self.__get_key(bucket_key).set_canned_acl(type)

    def get_file_to_ram(self):
        try:
            if not self.file:
                s = self.__get_key(self.key).get_contents_as_string()
                self.file = BytesIO(s)
            return self.file
        except Exception as e:
            exstr = traceback.format_exc()
            return False



    def push_file(self, filepath):
        if not filepath:
            return "filepath can not be None"
        bucket = self.__get_bucket(self.bucket_str)
        k = Key(bucket)
        k.key = self.key
        k.set_contents_from_filename(filepath, policy='public-read')
        # k.set_contents_from_file(filepath,policy='public-read')

    def delete_keys(self, bucket, keys):
        self.__get_bucket(bucket).delete_keys(keys)

    def close_connection(self):
        if self.conn:
            self.file = None
            self.conn.close()


_XFER_FILE = 'FILE'
_XFER_DIR = 'DIR'


class Xfer(object):
    '''''
    @note: upload local file or dirs recursively to ftp server
    '''

    def __init__(self):
        self.ftp = None

    def __del__(self):
        pass

    def setFtpParams(self, ip, uname, pwd, port=21, timeout=60):
        self.ip = ip
        self.uname = uname
        self.pwd = pwd
        self.port = port
        self.timeout = timeout

    def initEnv(self):
        if self.ftp is None:
            self.ftp = FTP()
            self.ftp.connect(self.ip, self.port, self.timeout)
            self.ftp.login(self.uname, self.pwd)

    def clearEnv(self):
        if self.ftp:
            self.ftp.close()
            self.ftp = None

    def uploadDir(self, localdir='./', remotedir='./'):
        if not os.path.isdir(localdir):
            return
        self.ftp.cwd(remotedir)
        for file in os.listdir(localdir):
            src = os.path.join(localdir, file)
            if os.path.isfile(src):
                self.uploadFile(src, file)
            elif os.path.isdir(src):
                try:
                    self.ftp.mkd(file)
                except:
                    sys.stderr.write('the dir is exists %s' % file)
                self.uploadDir(src, file)
        self.ftp.cwd('..')

    def uploadFile(self, localpath, remotepath='/'):
        if not os.path.isfile(localpath):
            return
        self.ftp.storbinary('STOR ' + remotepath, open(localpath, 'rb'))

    def __filetype(self, src):
        if os.path.isfile(src):
            index = src.rfind('\\')
            if index == -1:
                index = src.rfind('/')
            return _XFER_FILE, src[index + 1:]
        elif os.path.isdir(src):
            return _XFER_DIR, ''

    def upload(self, src):
        filetype, filename = self.__filetype(src)

        self.initEnv()
        if filetype == _XFER_DIR:
            self.srcDir = src
            self.uploadDir(self.srcDir)
        elif filetype == _XFER_FILE:
            self.uploadFile(src, filename)
        self.clearEnv()


FTP_COMAND = "lftp -c 'put %s -o ftp://%s:%s@%s/%s/'"
FTP_MKDIR_COMAND = "lftp -c 'mkdir ftp://%s:%s@%s/%s'"



class DateUtils(object):
    @staticmethod
    def __get_format(format):
        return format if format else "%Y%m%d"

    @staticmethod
    def str2date(date_str, format=None):
        assert date_str
        date = datetime.datetime.strptime(date_str,
                                          DateUtils.__get_format(format))
        return date

    @staticmethod
    def date2str(dateObj, format=None):
        assert dateObj and (isinstance(dateObj, datetime.date) or
                            time.struct_time)
        date_str = dateObj.strftime(DateUtils.__get_format(format))
        return date_str

    @staticmethod
    def get_current_time():
        # current_time=DateUtils.date2str(time.localtime(),"%Y%m%d %X")
        current_time = time.time()
        return int(current_time)

    @staticmethod
    def get_current_date():
        return datetime.datetime.fromtimestamp(DateUtils.get_current_time())

    @staticmethod
    def get_current_date_str():
        return DateUtils.date2str(DateUtils.get_current_date())

    @staticmethod
    def is_same_day(day1, day2):
        return day1 == day2

    @staticmethod
    def is_same_week(day1, day2):
        assert day1 and day2
        week1 = DateUtils.get_monday(day1)
        week2 = DateUtils.get_monday(day2)
        return week1 == week2

    @staticmethod
    def is_same_month(day1, day2):
        assert day1 and day2
        month1 = DateUtils.get_month_first_day(day1)
        month2 = DateUtils.get_month_first_day(day2)
        return month1 == month2

    @staticmethod
    def get_monday(day):
        assert day
        week = day.weekday()
        return DateUtils.plus_day(day, -week)

    @staticmethod
    def get_sunday(day):
        monday = DateUtils.get_monday(day)
        return DateUtils.plus_day(monday, 6)

    @staticmethod
    def get_month_first_day(day):
        month = day.month
        year = day.year
        return datetime.date(year, month, 1)

    @staticmethod
    def get_month_last_day(day):
        month = day.month
        year = day.year
        last_day = calendar.monthrange(year, month)[1]  # (星期,日期)
        return datetime.date(year, month, last_day)

    @staticmethod
    def get_next_monday(day):
        return DateUtils.plus_day(day, 7)

    @staticmethod
    def get_next_month(day):
        month_last_day = DateUtils.get_month_last_day(day)
        return DateUtils.plus_day(month_last_day, 1)

    @staticmethod
    def plus_day(day, num=None):
        assert day
        return day + datetime.timedelta(num if num or num == 0 else 1)

    @staticmethod
    def week_range(date):
        date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        year, week, week_day = date.isocalendar()
        start_date = week_day == 7 and date or date - datetime.timedelta(
                week_day)
        end_date = start_date + datetime.timedelta(6)
        return start_date, end_date

    @staticmethod
    def month_range(date):
        date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        return DateUtils.get_month_first_day(
                date), DateUtils.get_month_last_day(date)

    @staticmethod
    def get_local_time():
        # (tm_year=2016, tm_mon=8, tm_mday=11, tm_hour=10, tm_min=30, tm_sec=7, tm_wday=3, tm_yday=224, tm_isdst=0)
        # 包含所有的具体信息
        return time.localtime()

    @staticmethod
    def get_today_hour_time():
        return DateUtils.get_local_time()[3]



def move_to_trash(path,n):

    day = DateUtils.date2str(DateUtils.plus_day(DateUtils.get_current_date(), -n))

    trash_path = "/user/hdfs/.Trash/Current/%s/%s" % (path, day)

    cmd_create="hadoop fs -mkdir -p "+trash_path

    cmd = ("hadoop fs -mv %s/%s  " + trash_path) % (path, day)

    print(cmd_create)
    print(cmd)

    # os.system(cmd_create)
    # os.system(cmd)

s="""20171107
20171130
20171105
20171120
20171121
20171117
20171212
20171209
20180119
20171104
20171226
20171124
20171202
20180125
20171101
20171106
20171114
20171217
20171216
20171227
20171230
20180110
20180128
20180116
20180113
20171123
20171210
20171128
20171108
20171203
20180112
20180108
20180114
20171113
20171122
20171201
20171213
20171103
20171110
20171207
20180118
20180124
20171228
20180117
20171118
20180122
20171112
20171229
20180106
20171204
20171116
20171221
20171111
20171220
20171206
20171218
20180109
20180331
20171205
20180115
20171125
20180111
20180107
20180101
20180330
20171231
20171219
20180131
20180102
20180104
20171211
20171225
20180120
20171126
20171115
20180105
20171119
20171129
20180126
20180127
20171222
20171127
20180103
20171214
20171109
20171215
20171224
20171208
20180130
20180121
20180129
20180123
20171102
20180329
20171223
20180627
20180513
20180514"""

if __name__ == '__main__':


    # n = abs(int(sys.argv[1]))
    #
    #
    # delete_path1="/home/hdfs/dmp_ir/tag_label_map"
    # delete_path2="/home/hdfs/dmp_ir/label_result"
    #
    #
    # move_to_trash(delete_path1,n)
    # move_to_trash(delete_path2,n)

    split = s.split("\n")
    split=sorted(split)

    for ss in split:
        line="/home/hdfs/cdp/work/cdp.sh %s > /home/hdfs/cdp/work/log/%s.log" % (ss,ss)
        print(line)
