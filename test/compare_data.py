def main(before,last):

    before_data = file_content(before)
    last_data = file_content(last)
    before_dict={}
    for line in before_data:
        split = line.split(",")
        label_id=split[0]

        platform=split[1]
        uv=split[3]

        if platform == "4":
            before_dict.update({label_id:uv})

    write=open("/Users/tong/Desktop/label_difference.txt","w")

    for line in last_data:
        split = line.split(",")
        label_id = split[0]

        platform = split[1]
        uv = split[3]

        if platform=="4":
            before_uv = before_dict.get(label_id)
            if before_uv!=None:
                difference = int(uv) - int(before_uv)

                if difference != 0:
                    zhanbi = float(difference) / int(uv)
                    writeline = [label_id, uv, before_uv, str(difference), str(round(zhanbi, 3))+"\n"]
                    write.write(",".join(writeline))


def file_content(before):
    before_file = open(before, "r")
    readlines = before_file.readlines()
    return readlines


if __name__ == '__main__':

    main("/Users/tong/Desktop/LABEL20180218","/Users/tong/Desktop/LABEL20180218back")