import time

class TimerHelper:
    def __init__(self):
        self._start_at = None
        self._counter_start = None

    def get_current_time(self) -> int:
        return int(time.time())

    def init(self):
        self._start_at = self.get_current_time()

    def get_uptime(self) -> int:
        return self.get_current_time() - self._start_at

    def reset_timer(self):
        self._counter_start = self.get_current_time()

    def partial_timer(self) -> int:
        self._counter_end = self.get_current_time() - self._counter_start
        return self._counter_end