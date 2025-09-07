class EMEvaluator:
    NAME = "EM"

    @staticmethod
    def evaluate_one_pair(predict, reference):
        exact_match = 0
        if predict.split() == reference.split():
            exact_match = 1
        return exact_match
