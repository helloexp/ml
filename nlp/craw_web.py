# coding=utf-8

from bs4 import BeautifulSoup
import requests
import nltk
import re

if __name__ == '__main__':

    page = requests.get("https://www.python.org/")

    print(page)

    html = page.content
    soup = BeautifulSoup(html, 'html.parser')

    # print(soup)


    # print(soup.find_all('p')[0].get_text())
    # print(soup.find_all('p')[1].get_text())
    # print(soup.find_all('p')[2].get_text())
    # print(soup.find_all('p')[3].get_text())
    # print(soup.find_all('p')[4].get_text())
    print(html.split())

    print(re.split("\W+",str(html)))

    print(soup.get_text())














