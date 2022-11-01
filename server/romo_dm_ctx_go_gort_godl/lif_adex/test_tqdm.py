from tqdm import tqdm
import time
n = 5
m = 300


def t(m):
    for i2 in tqdm(range(m), colour='blue'):
        # do something, e.g. sleep
        time.sleep(0.01)
        pbar.update(1)
with tqdm(total=n * m, colour='red') as pbar:
    for i1 in tqdm(range(n)):
        t(m)