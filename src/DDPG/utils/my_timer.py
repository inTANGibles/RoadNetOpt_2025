import time
from typing import Optional


class MyTimer:
    __shared_data: Optional[dict] = None

    @classmethod
    def register_shared_data(cls, shared_data):
        """
        注册shared data
        """
        cls.__shared_data = shared_data

    def __init__(self, name: str, level: int):
        """
        :param name:
        :param level:
        """
        self.name = name
        self.level = level

    def __enter__(self):
        if MyTimer.__shared_data is None:
            return
        self.start_time = time.time()
        if 'time' not in MyTimer.__shared_data:
            MyTimer.__shared_data['time'] = {}
        if self.level not in MyTimer.__shared_data['time']:
            MyTimer.__shared_data['time'][self.level] = {}
        if self.name not in MyTimer.__shared_data['time'][self.level]:
            MyTimer.__shared_data['time'][self.level][self.name] = None

    def __exit__(self, exc_type, exc_value, _traceback):
        if MyTimer.__shared_data is None:
            return
        MyTimer.__shared_data['time'][self.level][self.name] = time.time() - self.start_time
