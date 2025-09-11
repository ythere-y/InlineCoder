from fuzzywuzzy import fuzz


class ESEvaluator:
    NAME = "ES"

    @staticmethod
    def evaluate_one_pair(predict, reference):
        edit_sim = fuzz.ratio(predict, reference)
        return edit_sim
