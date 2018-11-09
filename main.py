import os
import time
import re

'''
Useful links:
https://steamcommunity.com/sharedfiles/filedetails/?id=309868072
https://stackoverflow.com/questions/5419888/reading-from-a-frequently-updated-file
'''

def follow(thefile):
    thefile.seek(0,2)
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line

dir_path = os.path.dirname(os.path.realpath(__file__))
log_file_name = "test.txt"
file_path = os.path.join(dir_path, log_file_name)

log_file = open(file_path, "r")
loglines = follow(log_file)

pattern_type = re.compile(r"type: (\d+)")
for line in loglines:
    temp = pattern_type.findall(line)
    if temp != []:
        print(temp[0])