import ssl
import urllib.request

first_url = "https://www.douban.com/"


webheader = {
    'Accept': 'text/html, application/xhtml+xml, */*',
    # 'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-CN',
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
    'DNT': '1',
    'Connection': 'Keep-Alive',
    'Host': 'www.douban.com'
}

context = ssl._create_unverified_context()
req = urllib.request.Request(url=first_url, headers=webheader)
webPage = urllib.request.urlopen(req,context=context)
data = webPage.read().decode('utf-8')

print(data)
print(type(webPage))
print(webPage.geturl())
print(webPage.info())
