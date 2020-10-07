import time

from toolkit.logger import logger

class Timer:
    def __init__(self) -> None:
        super().__init__()
        self.s_time = time.time()

    def stop(self):
        return time.time() - self.s_time

def timed(method):
    def timed_fun(*args, **kw):
        timer = Timer()
        result = method(*args, **kw)
        duration = timer.stop()
        logger.info("===> {:5.3f}s for <{}>".format(duration, method.__name__))
        return result
    return timed_fun