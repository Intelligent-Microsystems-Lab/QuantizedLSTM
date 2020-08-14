import glob, os, argparse, re, hashlib, uuid, collections, math, time, random

for count, filename in enumerate(os.listdir("cough")):
    if filename[0] == ".":
        continue
    print(filename)
    dst = "cough/" + ('%08x' % random.randrange(16**8)) + "_nohash_" + filename.replace(" ", "_")
    os.rename("cough/"+filename, dst)