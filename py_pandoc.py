import pypandoc
import os
#os.environ.setdefault('PYPANDOC_PANDOC', 'D:\Software\pandoc-2.5-windows-x86_64\pandoc')
#os.environ.setdefault('PYPANDOC_PANDOC', 'C:\Program Files\MiKTeX 2.9\miktex\bin\x64')

def main():
    marks_down_links = {
        "Standford CS231n 2017 Summary":
            "https://raw.githubusercontent.com/mbadry1/CS231n-2017-Summary/master/README.md",
    }

    # Extracting pandoc version
    print("pandoc_version:", pypandoc.get_pandoc_version())
    print("pandoc_path:", pypandoc.get_pandoc_path())
    print("\n")

    # Starting downloading and converting
    for key, value in marks_down_links.items():
        print("Converting", key)
        pypandoc.convert_file(
            value,
            'pdf',
            extra_args=['--pdf-engine=pdflatex', '-V', 'geometry:margin=1.5cm'],
            outputfile=(key + ".pdf")
        )
        print("Converting", key, "completed")


if __name__ == "__main__":
    main()