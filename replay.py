from snake_deep_q import *

import torch as T
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("Must supply path for state dict")
    replay = Replay()
    replay.update_data(T.load(sys.argv[1]), "NA")
    replay.start_sim()
    replay.thread.join()