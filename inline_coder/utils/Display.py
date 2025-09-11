from rich import print


class TableMaker:
    @staticmethod
    def make_table(head_line, data_lines):
        table_str = ""
        table_str += "|" + "|".join(head_line) + "|\n"
        table_str += "|" + ":---|" * len(head_line) + "\n"
        for line in data_lines:
            table_str += "|" + "|".join(line) + "|\n"
        return table_str


class PanelDisplay:
    color_list = ["red", "green", "blue", "yellow", "magenta", "cyan", "white"]

    @staticmethod
    def display_panels(contents, titles):
        from rich.panel import Panel
        from rich.console import Console
        from rich.columns import Columns

        panel_list = []
        console = Console()
        panel_width = console.width // len(contents) - 2
        for content, title in zip(contents, titles):
            panel = Panel(
                content,
                title=title,
                width=panel_width,
                style=PanelDisplay.color_list[
                    len(panel_list) % len(PanelDisplay.color_list)
                ],
            )
            panel_list.append(panel)
        columns = Columns(panel_list, equal=True)
        console.print(columns)


class HistDisplay:
    @staticmethod
    def display_histogram(
        data: list, title: str, xlabel: str, ylabel: str, figure_path
    ):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=30, color="blue", alpha=0.7)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(axis="y", alpha=0.75)
        plt.savefig(figure_path)
        plt.close()
        print(f"BLEU score difference histogram saved to {figure_path}")
