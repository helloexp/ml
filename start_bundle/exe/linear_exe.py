
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-echo")
parser.add_argument("date")
args = parser.parse_args()

print(args)

