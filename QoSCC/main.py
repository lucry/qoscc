import random
import argparse
import numpy as np
import tensorflow as tf

from py_agent import *

def add(a, b):
    print("start add function in def...")
    return a+b

class add_class:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def add(self, other):
        print("start add function in class...")
        return self.a+self.b

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', type=str, choices=['learner', 'actor'], required=True,
                        help='Job name: either {\'learner\', actor}')
    parser.add_argument('--task', type=int, required=True, help='Task id')
    config = parser.parse_args()
    jobname = config.job_name
    task = config.task

    drl = PY_Agent(jobname, task)
    drl.set_target(100, 0.4, 10, 0.3)
    drl.store_experience()
    for i in range(100):
        max_thr = random.randint(0, 100)
        min_rtt = random.randint(0, 10)
        last_loss = random.random()
        # drl.train()
        a = drl.get_action(max_thr, min_rtt)
        print(a[0][0])



