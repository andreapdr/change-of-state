from collections import Counter
from datamanager.dataloader import load_processed_dataset, load_cos_verbs


def balance_dataset(dataset, cos_verbs, delta=0.1):
    """
    Balance dataset to contain approximately (delta) same number distribution of cos verbs in the true_caption and foiled_caption.
    """
    merged_splits = {k: v for d in dataset for k, v in d.items()}
    counter_actual_verbs = Counter([v["verb"] for v in merged_splits.values()])
    # TODO: the mapping is an Injective function - but we could have more cos verb mapping to multiple reverse verbs (use get_synonyms)
    _reverse2action_mapping = {
        v["state-inverse"]: k
        for k, v in cos_verbs.items()
        if k in counter_actual_verbs.keys()
    }  # TODO: state-inverse key is not fully appropritate naming
    counter_reversed_action = {
        k: counter_actual_verbs.get(k, 0) for k in _reverse2action_mapping.keys()
    }

    _action2reverse_mapping = {
        k: cos_verbs[k]["state-inverse"] for k in counter_actual_verbs.keys()
    }

    action_mapping = {}
    action_mapping.update(_reverse2action_mapping)
    action_mapping.update(_action2reverse_mapping)

    to_sample = _to_sample(
        action_mapping, counter_actual_verbs, counter_reversed_action
    )

    # TODO: remove this print
    for k, v in to_sample.items():
        if v > 0:
            print(f"{k}: {v}")


def _to_sample(action_mapping, counter_actual_verbs, counter_reverse_action):
    # TODO: this logic is a mess
    to_sample = {}
    for verb in action_mapping.keys():
        reverse_verb = action_mapping[
            verb
        ]  # TODO: e.g., to dirty -> to wash but also to wash -> to stain / to dirty
        actual_count = counter_actual_verbs[verb]
        target_count = counter_reverse_action.get(reverse_verb, 0)
        n_samples = min(actual_count, target_count)
        to_sample[verb] = n_samples
        to_sample[reverse_verb] = n_samples
    return to_sample


if __name__ == "__main__":
    coin = load_processed_dataset("coin", level=3, model_size="trf", max_captions=None)
    ikea = load_processed_dataset("ikea", level=3, model_size="trf", max_captions=None)
    rareact = load_processed_dataset(
        "rareact", level=3, model_size="trf", max_captions=None
    )
    # smsm = load_processed_dataset(
    #     "smsm", level=3, model_size="trf", max_captions=None
    # )
    star = load_processed_dataset("star", level=3, model_size="trf", max_captions=None)
    yc = load_processed_dataset("yc", level=3, model_size="trf", max_captions=None)

    merged_dataset = [{}, {}, {}]
    for data in [ikea, rareact, star, yc]:
        for i, split in enumerate(data):
            merged_dataset[i].update(split)

    cos_mapping = load_cos_verbs("triggers/chatgpt_cos.csv", augment_it=True)
    balance_dataset(merged_dataset, cos_mapping, delta=0.1)
