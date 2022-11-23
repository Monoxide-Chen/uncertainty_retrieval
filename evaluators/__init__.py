from evaluators.tirg_evaluator import SimpleEvaluator


def get_evaluator_cls(configs):
    evaluator_code = configs['evaluator']
    if evaluator_code == 'simple':
        return SimpleEvaluator
    else:
        raise ValueError("There's no evaluator that has {} as a code".format(evaluator_code))
