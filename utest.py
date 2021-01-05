failed = 0
total = 0
# from models.nets import test
# print("testing models/nets.py")  
# try:
#     total += 1
#     test()
# except Exception as e:
#     print(f'models/nets.py failed: {e}')
#     failed += 1

from utils.datastream import ZeroShotDataCollection
zsdc = ZeroShotDataCollection("data")

print(f"passed {total - failed} / {total}")