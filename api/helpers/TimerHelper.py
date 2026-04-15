import time

class TimerHelper:
    def __init__(self):
        self._start_at = None
        self._counter_start = None

    def init(self):
        self._start_at = time.time()

    def get_uptime(self) -> int:
        return time.time() - self._start_at

    def reset_timer(self):
        self._counter_start = time.time()

    def partial_timer(self) -> int:
        self._counter_end = time.time() - self._counter_start
        return self._counter_end