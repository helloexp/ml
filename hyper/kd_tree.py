# -*- coding:utf-8 -*-

X = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
# X= [(30,40), (5,25), (10,12), (70,70), (50,30), (35,45)]

class Node(object):
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls, *args, **kwargs)

    def __str__(self):
        return {"value": self.value, "left": self.left, "right": self.right}


class KDTree(object):
    def __init__(self, X):
        self.k = len(X[0])
        self.root = None
        self.build(X)

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls, *args, **kwargs)

    def get_index(self, deep):
        return deep % self.k

    def build(self, X, deep=0):

        if len(X) <= 0:
            return None

        index = self.get_index(deep)

        index_arr = sorted(map(lambda x: x[index], X))
        X.sort(key=lambda x: x[index])

        # median_index = int(math.ceil(float(len(index_arr)) / 2))-1
        median_index = len(index_arr) // 2

        node = Node(X[median_index])

        if not self.root:
            self.root = node

        left = X[0:median_index]

        node.left = self.build(left, deep + 1)

        if (median_index < len(X)):
            right = X[median_index + 1:]
            node.right = self.build(right, deep + 1)

        return node

    def query(self, node):

        pass


def iter_tree(node):
    if node == None:
        return ""
    v=node.value
    s={"value": v}
    left=iter_tree(node.left)
    s.update({"left":left})
    right=iter_tree(node.right)
    s.update({"right":right})
    return s



if __name__ == '__main__':
    kd_tree = KDTree(X)

    print iter_tree(kd_tree.root)
