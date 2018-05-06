#coding=utf8
import itchat

def test_itchat():

    itchat.auto_login()

    # itchat.send('Hello, filehelper', toUserName='filehelper')

    friends = itchat.search_friends()
    chatrooms = itchat.get_chatrooms()

    chatroom=chatrooms[0]


    itchat.logout

    print friends

@itchat.msg_register(itchat.content.TEXT)
def reply_text(msg):
    return msg["text"]

if __name__ == '__main__':


    test_itchat()