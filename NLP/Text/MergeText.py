# basic merge script for text files
import glob

# select files with .txt suffix
read_files = glob.glob("*.txt")

# loop until all documents merged to result.txt
with open("result.txt", "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())
