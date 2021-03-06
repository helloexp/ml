import os

import PyPDF2
import urllib3

uri = "http://mml-book.github.io/book/chapter%s.pdf"
base_file = "./chapter%s.pdf"


def craw():
    http = urllib3.PoolManager()

    for i in range(1, 13):
        if (i < 10):
            i = "0" + str(i)
        file_name = base_file % str(i)
        file_pdf = open(file_name, "wb+")
        crawl_uri = uri % str(i)
        result = http.request("GET", crawl_uri)
        print(result.data)

        file_pdf.write(result.data)

        file_pdf.close()


def get_all_pdf_files(param):
    return list(filter(lambda f: f.endswith("pdf"), param))


def main():
    all_pdfs = reversed(get_all_pdf_files(os.listdir('./')))
    print(all_pdfs)
    if not all_pdfs:
        raise SystemExit('No pdf file found!')

    dst_pdf = PyPDF2.PdfFileWriter()

    page = 0
    for i in range(1, 13):

        if (i < 10):
            i = "0" + str(i)

        f = base_file % str(i)

        reader = PyPDF2.PdfFileReader(f)
        pages = reader.getNumPages()

        for p in range(0, pages):
            dst_pdf.addPage(reader.getPage(p))

        dst_pdf.addBookmark("ch" + str(i), page)

        page = page + pages

    file_io = open("./merge_pdf.pdf", 'wb')
    dst_pdf.write(file_io)

    file_io.close()


if __name__ == '__main__':
    # craw()
    main()
