import time

def timer(func):
    def timed(*args, **kwargs):
        start_time = time.time()
        res = func(*args)
        end_time = time.time()
        delta_t = end_time - start_time
        print(f'{func.__name__} took {delta_t:.3f} seconds to run')

        return res
    return timed
