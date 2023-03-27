from functools import partial

from model import get_conditional_frequency, get_conditional_probability, get_frequency, load_frequency_dict, load_prefix_frequency_dict, load_probabilistic_model, load_vocab

def laplace_smoothing(conditional_frequency_dict, prefix_frequency_dict, token, pinyin=None, prefix="", vocab_num=6763):
    """
        laplace smoothing
        return p(token|prefix)
    """
    probability = (get_conditional_frequency(conditional_frequency_dict, prefix, token) + 1) / (get_frequency(prefix_frequency_dict, prefix) + vocab_num)
    return probability


def interpolation_smoothing(conditional_probabilistic_dict, token, para_lambda, pinyin=None, prefix=""):
    """
        interpolation smoothing
        return p(token|prefix)
    """
    # if len(prefix) > 1:
    #     return get_conditional_probability(probabilistic_dict, prefix, token) * (1 - para_lambda) + smoothing_probability(probabilistic_dict, prefix[-1:], token) * para_lambda
    probability = get_conditional_probability(conditional_probabilistic_dict, prefix, token) * (1 - para_lambda) + get_conditional_probability(conditional_probabilistic_dict, "", token) * para_lambda
    return probability


def normalized_probability(pinyin_dict, probability_fn, token, pinyin, prefix=""):
    """
        normalized probability
        return p(token|prefix, pinyin)
    """
    probability = probability_fn(token=token, prefix=prefix)
    if probability == 0:
        return 0
    totall_probability = sum([probability_fn(token=token, prefix=prefix) for token in pinyin_dict[pinyin]])
    return probability / totall_probability

def get_probability_function(probability_type="interpolation", para_lambda=0.03, normalized=False):
    vocabs, pinyin_dict = load_vocab()

    if probability_type == "laplace":
        conditional_frequency_dict = load_frequency_dict(vocabs=vocabs)
        prefix_frequency_dict = load_prefix_frequency_dict(conditional_frequency_dict)
        probability_fn = partial(
            laplace_smoothing,
            conditional_frequency_dict=conditional_frequency_dict,
            prefix_frequency_dict=prefix_frequency_dict,
            vocab_num=len(vocabs)
        )
    else:
        conditional_probabilistic_dict = load_probabilistic_model(vocabs=vocabs)
        probability_fn = partial(
            interpolation_smoothing,
            para_lambda=para_lambda,
            conditional_probabilistic_dict=conditional_probabilistic_dict
        )

    if normalized:
        return partial(
            normalized_probability,
            pinyin_dict=pinyin_dict,
            probability_fn=probability_fn
        )

    return probability_fn