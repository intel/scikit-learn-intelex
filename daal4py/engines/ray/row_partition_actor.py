import pandas


class RowPartitionsActor:
    def __init__(self, node):
        self.row_parts_ = []
        self.node_ = node

    def set_row_parts(self, *row_parts):
        self.row_parts_ = pandas.concat(list(row_parts), axis=0)

    def set_row_parts_list(self, *row_parts_list):
        self.row_parts_ = list(row_parts_list)

    def append_row_part(self, row_part):
        self.row_parts_.append(row_part)

    def concat_row_parts(self):
        self.row_parts_ = pandas.concat(self.row_parts_, axis=0)

    def get_row_parts(self):
        return self.row_parts_

    def get_actor_ip(self):
        return self.node_

