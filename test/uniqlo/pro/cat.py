import os

files = os.listdir("./")

format="cat > hbase-site.xml << EOF\n%s\nEOF"

t = open("uniqlo_pro.txt","w")
for f in files:
    if f.endswith("xml"):
        xml = open("./" + f,"r")
        content = xml.readlines()
        write=""
        for l in content:
            write=write+l
        write = format % write
        t.write(write)

t.close()















