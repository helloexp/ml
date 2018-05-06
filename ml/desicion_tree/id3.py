# -*- coding:utf-8 -*-
import csv

import math




class Passanger(object):


    def __init__(self,survial,pc_class,name,sex,age,fare):
        self.survial=survial
        self.pc_class=pc_class
        self.name=name
        self.sex=self.set_sex(sex)  #
        self.age=age
        self.age_stage(age)
        self.fare=fare #待处理

    def x(self):
        return [self.pc_class, self.sex, self.age_stage]

    def y(self):
        return self.survial

    def set_sex(self,sex):
        if(sex=="male"):
            return 0
        else:
            return 1


    def train_x_y(self):
        return [self.pc_class,self.sex,self.age_stage,self.survial]

    def age_stage(self,age):
        age = float(age)
        if(age<=18):
            self.age_stage="1"
        elif(age<=24):
            self.age_stage="2"
        elif(age<=30):
            self.age_stage="3"
        elif(age<=35):
            self.age_stage="4"
        elif(age<=40):
            self.age_stage="5"
        else:
            self.age_stage="6"

    @staticmethod
    def build(line):
        line_data = [line[0], line[1], line[2], line[3], line[4], line[5]]
        for ld in line_data:
            if ld=="":
                return None
        passanger = Passanger(*line_data)
        return passanger

    def __str__(self):
        return reduce(lambda x,y:str(x)+" "+str(y),[self.name,self.pc_class,self.sex,self.age_stage,self.fare,self.survial])


def information_formula(value_size,size):
    p = float(value_size) / size
    return -p * math.log(p,2)

def information(input_list):
    distinct=set(input_list)

    size = len(input_list)

    information_res=0

    for d in distinct:
        d_size = input_list.count(d)
        information_res+=information_formula(d_size,size)

    return information_res


def split(will_split_data):

    data_y_list={}

    for data in will_split_data:
        feature_y = data_y_list.get(data[0], [])
        if(not feature_y):
            data_y_list.update({data[0]:feature_y})

        feature_y.append(data[1])

    return data_y_list


def calc_sub_tree_information(data_y_list,p_add_n):

    res=0
    for data,y in data_y_list.iteritems():
        i_expect_information = float(len(y)) / p_add_n * information(y)
        res+=i_expect_information

    return res


def get_split_feature(feature_gain):

    s = sorted(feature_gain.items(), lambda x, y: cmp(y[1], x[1]))

    if len(s)>0:
        return s[0][0]
    else:
        return None


def split_tree(X, Y, split_feature):

    res={}

    for x,y in zip(X,Y):
        value = x[split_feature]
        new_xy = res.get(value)
        if(not new_xy):
            new_xy=[]
            res.update({value:new_xy})

        feature_add_one = split_feature + 1
        if(feature_add_one>=len(X)):
            feature_add_one=len(x)

        delete_feature_data=[]
        delete_feature_data.extend(x[0:split_feature])
        delete_feature_data.extend(x[feature_add_one:])
        new_xy.append([delete_feature_data,y])
    return res


def split_to_x_y(x_y):
    X=[]
    Y=[]
    for l in x_y:
        X.append(l[0])
        Y.append(l[1])
    return X,Y


def train_with_dt(X, Y):
    sub_trees={}
    info = information(Y)

    feature_num=len(X[0])

    if(feature_num==0):
        return Y[0]

    feature_gain={}

    for i in range(feature_num):
        feature_data = map(lambda x: x[i], X)
        will_split_data = zip(feature_data, Y)
        data_y_list = split(will_split_data)
        tree_information = calc_sub_tree_information(data_y_list, len(will_split_data))
        information_gain=info-tree_information
        feature_gain.update({i:information_gain})

    split_feature = get_split_feature(feature_gain)

    sub_trees = split_tree(X, Y, split_feature)

    for feature,x_y in sub_trees.iteritems():
        sub_X,sub_y = split_to_x_y(x_y)
        print "sub_x",sub_X
        print "sub_y",sub_y
        print "train re"
        sub_trees.update({feature:train_with_dt(sub_X,sub_y)})

    print "sub_trees",sub_trees
    print "split_feature:",split_feature
    print "information:",info
    print "feature_num:",feature_num
    print "feature_gain:",feature_gain

    return sub_trees


def titanic_data():
    global titanic_x, titanic_y
    titanic = csv.reader(open("../resource/descion_tree.csv"))
    titanic_passanger = map(lambda x: Passanger.build(x), titanic)
    titanic_x = []
    titanic_y = []
    for tp in titanic_passanger:
        if (tp):
            titanic_x.append(tp.x())
            titanic_y.append(tp.y())
    return titanic_x,titanic_y


if __name__ == '__main__':
    titanic_x,titanic_y=titanic_data()

    print len(titanic_x),titanic_x
    print len(titanic_y),titanic_y

    #print information([1,2,3,3,2,1,4,4])   #test information

    print  train_with_dt(titanic_x,titanic_y)




    # sub_trees= split_tree([[1,2,3],[1,3,4],[2,3,3]],[0,1,2],0)
    # print sub_trees
    # for feature,x_y in sub_trees.iteritems():
    #     print split_to_x_y(x_y)
    #



