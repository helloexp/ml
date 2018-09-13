#coding:utf-8


# title,authors,groups,keywords,topics,abstract
import numpy as np
import pandas as pd

paper_file="../resource/clusting/paper.txt"


def cluster_paper():
    paper_lines = pd.read_csv(paper_file)
    keys = paper_lines.keys()
    title = keys.values
    K = 1
    title_column = paper_lines["title"].map(lambda x: str(x).lower())
    author_column = paper_lines["authors"].map(lambda x: str(x).lower())
    groups_column = paper_lines["groups"].map(lambda x: str(x).lower())


# cluster_paper()





def generate_shingles(k,line):
    res=set()

    if(k<=0):
        return res

    line_split = line.split(" ")

    for i in range(0,len(line_split)):

        next_k=i+k

        if next_k>len(line_split):
            break

        join=line_split[i]
        for j in range(i+1,next_k):
            join=join+" "+line_split[j]

        res.add(join)

    return res

