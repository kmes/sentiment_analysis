import time

class TimerHelper:
    def __init__(self):
        self._start_at: float = None
        self._counter_start: float = None
        self._counter_end: float = None

    def get_current_time(self) -> float:
        """
        Returns the current time in float with milliseconds precision.
        """
        return time.perf_counter()

    def init(self):
        """
        Initializes the timer and set the start time to the current time in float with milliseconds precision.
        """
        self._start_at = self.get_current_time()

    def get_uptime(self) -> int:
        """
        Returns the time elapsed since the last call to init() in int milliseconds.
        """
        return int((self.get_current_time() - self._start_at) * 1000)

    def reset_timer(self):
        """
        Resets the timer and set the counter start time to the current time in float with milliseconds precision.
        """
        self._counter_start = self.get_current_time()

    def partial_timer(self) -> int:
        """
        Returns the time elapsed since the last call to reset_timer() in int milliseconds.
        """
        self._counter_end = self.get_current_time() - self._counter_start
        return int(self._counter_end * 1000)