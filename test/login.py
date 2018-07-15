# ssh hypers-tong.yue@bastion.uniqlo.cn
# hypers
# !Icl2mhOThy

import os
# -*- coding: utf-8 -*-

import pexpect

host="hypers-tong.yue@bastion.uniqlo.cn"
passwd="Yue123258!"


ssh = pexpect.spawn("ssh "+host)

i = ssh.expect(["Password: ", "continue connecting (yes/no)?"], timeout=10)

if i == 0 :
    ssh.sendline(passwd)
elif i == 1:
    ssh.sendline('yes\n')
    ssh.expect('password: ')
    ssh.sendline(passwd)
ret = 0






