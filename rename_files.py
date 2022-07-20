import glob
import re
import os

def filter_files_expression(special_char):
    return f"(_\d_\d-{special_char}\d_)"

def change_given_files(allfilenames, special_char, replace_char=""):
    regex_str = filter_files_expression(special_char)
    ref = lambda file: re.search(regex_str, file)
    rexfiles =  list(filter(ref, allfilenames))

    for file in rexfiles:
        new_file_name = re.sub(special_char, replace_char, file)
        os.rename(file, new_file_name)

def change_filenames(special_char=">"):
    # bash command
    os.system("""find $pwd -type d | awk -F"/" 'NF > max {max = NF} END {print max}' >> filestructure.txt""")
    max_dir_depth = open("filestructure.txt", "r").read().split("\n")[0]
    max_dir_depth = int(max_dir_depth)
    for i in range(max_dir_depth):
        allfilenames = glob.glob("**/"*(max_dir_depth-i)+"*")
        change_given_files(allfilenames, special_char)
if __name__ == "__main__":
    change_filenames()
