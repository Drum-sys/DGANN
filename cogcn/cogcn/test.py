# from ast import Global
# from pathlib import Path
# import glob
# dataset_base = Path("/c/python/Mono2Micro-FSE-2021/datasets_runtime1/")
# # print(dataset_base.glob("*"))
# # globalPath = dataset_base.glob("*")
# for path in  glob.glob("/c/python/Mono2Micro-FSE-2021/datasets_runtime1/*"):
#     print(path)
#     print("ljw ")
    
# for path in  glob.glob("C:\\python\\Mono2Micro-FSE-2021\\datasets_runtime1\\*"):
#     print(path)
#     print("ljwp")
    

# # for project_dir in dataset_base.glob("*"):
# #     print(project_dir)
# #     print(project_dir)

from tabulate import tabulate
from time import time
import pandas as pd
header = ["Partitions", "BCS[-]"]
res = [[1, 2.34], [-56, "8.999"], ["2", "10001"]]
# print(tabulate(res, header, tablefmt="tsv"))
test = pd.DataFrame(columns=header, data=res)

res_all = []
header = ["Partitions", "BCS[-]"]

for i in range(2):
    
    res = [[1, 2.34]]
    res_all.append(res[0])
test = pd.DataFrame(columns=header, data=res_all)
# print(test)


str = "C:\\python\\icse-deeply\\additional-files\\output\\deeply-lossless\\GAT\\" + "slurm-" + str(time()) + ".csv"
test.to_csv(str)
print(str)