class MultiColumn:
    DEFAULT_REGEX = 'c' # 'c' or 'l' or 'r'
    def __init__(self, 
                 content: str, 
                 row_num: int=1, 
                 regex: str=DEFAULT_REGEX):
        self.row_num = row_num
        self.regex = regex
        self.content = content

class LatexTable:
    def __init__(self, 
                 caption: str, 
                 label: str,
                 format: str,
                 column_num: int = None,
                 width: int=1.0, 
                 ):
        self.caption = caption
        self.label = label
        self.width = width
        self.indent_level = 0
        self._contents: list[str] = ["\n"]
        self._column_num = column_num
        self._outputs: callable = [
            self._header,
            self._spacer_up,
            self._tabular_start(format),
            self._spacer_up,
            self.contents,
            self._spacer_down,
            self._tabular_end,
            self._spacer_down,
            self._footer
            ]

    def compile(self) -> str:
        """Alias of __str__"""
        return self.__str__()

    def contents(self) -> str:
        return self._spacer().join(self._contents)

    def add_columns(self, columns: list):
        # Check column number
        if self._column_num is not None:
            _cnt = 0
            for column in columns:
                if isinstance(column, MultiColumn):
                    _cnt += column.row_num
                else:
                    _cnt += 1
            if _cnt != self._column_num:
                raise ValueError("Column number mismatch. Expected: {0}, Actual: {1}".format(self._column_num, _cnt))

        _content = []
        for column in columns:
            if isinstance(column, MultiColumn):
                _content.append(r"\multicolumn{{{row_num}}}{{{regex}}}{{{content}}}".format(
                    row_num=column.row_num,
                    regex=column.regex,
                    content=column.content
                ))
            else:
                _content.append(str(column))
            _content.append("&")
        _content.pop() # Remove last "&"
        _content.append(r"\\" + "\n")
        self._contents.append("".join(_content))

    def add_hline(self):
        self._contents.append(self._hline())

    def _spacer(self):
        return "    " * self.indent_level

    def _spacer_up(self):
        self.indent_level += 1
        return "    " * self.indent_level

    def _spacer_down(self):
        self.indent_level -= 1
        return "    " * self.indent_level

    def _header(self):
        return r"""\begin{{figure}}[ht]
    \centering
    \caption{{{caption}}}
    \label{{fig:{label}}}
""".format(caption=self.caption, label=self.label, width=self.width)

    def _tabular_start(self, format: str):
        def _f():
            return r"\begin{{tabular}}{{{format}}}".format(format=format) + "\n"
        return _f

    def _tabular_end(self):
        return "\\end{tabular}\n"

    def _footer(self):
        return "\\end{figure}\n"

    def _bf(self, text: str):
        return r"\textbf{{{text}}}".format(text=text)

    def _hline(self):
        return "\\hline\n"

    def __str__(self):
        return "".join([func() for func in self._outputs])


if __name__ == '__main__':
    table = LatexTable(
        caption="Caption",
        label="label",
        format=r"l|*{4}{r}|*{4}{r}",
        column_num=9,
        width=0.9,
    )
    table.add_hline()
    table.add_columns(["optuna", MultiColumn("False", 8)])
    table.add_columns(["preprocess", MultiColumn("False", 8)])
    table.add_columns([r"ae\_preprocess", MultiColumn("False", 8)])
    table.add_columns([
        MultiColumn('layers', 1, 'l|'),
        "none",
        [20, 10, 5],
        [20, 15, 10],
        [20, 15, 10, 5],
        "none",
        [20, 10, 5],
        [20, 15, 10],
        [20, 15, 10, 5],
    ])
    table.add_hline()
    table.add_columns([
        MultiColumn("Dataset", regex='l|'),
        MultiColumn("minority F-accuracy", 4, 'c|'),
        MultiColumn("macro F-accuracy", 4),
    ])
    table.add_hline()
    print(table.compile())