import numpy as np

class SlidingMin:
    def __init__(self, window_size=int(1e5)):
        self.window_size = int(window_size)
        self.arr = np.ones((self.window_size))*np.inf
        self.ct = 0
        self.cur_min_idx = None
        self.cur_min_val = None

    def insert(self, x):
        self.arr[self.ct] = x
        self.ct = (self.ct + 1) % self.window_size

        if (self.cur_min_val is None) or (self.cur_min_val > x):
            self.cur_min_idx = (self.ct - 1) % self.window_size
            self.cur_min_val = x
        else:
            if (self.cur_min_val < x) and (self.cur_min_idx) == (self.ct - 1) % self.window_size:
                self.cur_min_idx = np.argmin(self.arr)
                self.cur_min_val = self.arr[self.cur_min_idx]

    def get_min(self):
        return self.cur_min_val

if __name__ == '__main__':
    window_size = int(1e5)
    buffer = SlidingMin(window_size)

    a = np.random.rand((int(1e6)))


    import time

    t_a = time.time()
    for i in range(a.shape[0]):
        buffer.insert(a[i])
        tmp = np.min(a[max(-window_size + i, 0):i+1])    
        tmpa = buffer.get_min()
        print(tmp, tmpa)
        if (i+1) % 10000 == 0:
            t_b = time.time()
            #print(i, (t_b - t_a) / (i+1), t_b - t_a)
    

    # for i in range(a.shape[0]):
    #     buffer.insert(a[i])
    #     if (i+1) % 10000 == 0:
    #         t_b = time.time()
    #         print(i, (t_b - t_a) / (i+1))