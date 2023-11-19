# Just a very basic implementation of 
# "Maybe"-like monad to pass entities
# across different types of converters

class Maybe:
    def __init__(self, body = None):
        self.body = body

    @property
    def has_value(self) -> bool:
        return self.body is not None

    def bind(self, func, *args, **kwargs):
        not_empty = func is not None
        if self.has_value and not_empty:
            self.body = func(*args, **kwargs)
        return self

    def unwrap(self):
        return self.body

    def unsafe_unwrap(self, msg: str = "No value"):
        if not self.has_value:
            raise ValueError(msg)
        return self.unwrap()
