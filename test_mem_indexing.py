import os
import re
import time
import ctypes

def get_avail_mem_gb():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullAvailPhys / (1024**3)

print(f"Initial available RAM: {get_avail_mem_gb():.2f} GB")
train_dir = r"c:\Users\Shourya\Desktop\dataset\student_resource\dataset\train"

# Test streaming 1,000,000 lines of S2
t0 = time.time()
LEGAL = {'pvt', 'ltd', 'private', 'limited', 'inc', 'corp', 'corporation', 'llc', 'llp', 'co', 'company'}

# Compact index: key -> list of integer IDs (to save RAM)
from collections import defaultdict
index = defaultdict(list)
count = 0

with open(os.path.join(train_dir, "train_source2.tsv"), "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        count += 1
        p = line.strip().split('\t')
        if len(p) >= 4:
            eid = p[0] # e.g. S2-12345
            # store integer ID to use 8 bytes instead of 60 bytes string
            int_id = int(eid[3:])
            name = p[1].lower()
            # first word
            toks = [t for t in re.sub(r'[^a-zA-Z0-9]', ' ', name).split() if len(t) >= 3 and t not in LEGAL]
            if toks:
                index[(p[3], toks[0])].append(int_id)
        if count >= 1000000:
            break

t1 = time.time()
print(f"Processed 1,000,000 rows in {t1 - t0:.2f}s ({1000000/(t1-t0):.0f} rows/s)")
print(f"Total unique keys: {len(index)}")
print(f"Available RAM after 1M rows: {get_avail_mem_gb():.2f} GB")
