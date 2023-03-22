import json
import pandas as pd

from collections import Counter, defaultdict
from datamanager.dataloader import load_cos_verbs, load_processed_dataset


def get_balanced_verb_distribution(dataset, cos_verbs, delta=0.1):
    merged_splits = {k: v for d in dataset for k, v in d.items()}
    hypernym_count = Counter([v["verb-hypernym"] for v in merged_splits.values()])

    sorted_hyp_keys = sorted(hypernym_count.keys())
    visited = set()
    for h in sorted_hyp_keys:
        h_count = hypernym_count[h]
        state_inverse = cos_verbs[h]["state-inverse"]
        if h in visited:
            continue
        inv_count = hypernym_count[state_inverse]
        min_count = min(h_count, inv_count)
        print(f"{h}: {h_count}\t{state_inverse}: {inv_count}\tmin: {min_count}")
        verb_set = set(
            v["verb"] for v in merged_splits.values() if v["verb-hypernym"] == h
        )
        inv_set = set(
            v["verb"]
            for v in merged_splits.values()
            if v["verb-hypernym"] == state_inverse
        )
        print(f"- verb_set: {sorted(list(verb_set))}")
        print(f"- inv_set: {sorted(list(inv_set))}")
        visited.add(state_inverse)


def balance_dataset(dataset, cos_verbs, delta=0.1, max_verbs=50, verbose=False):
    """
    Balance dataset to contain approximately (delta) same number distribution of cos verbs in the true_caption and foiled_caption.
    """
    merged_splits = {k: v for d in dataset for k, v in d.items()}
    hypernym_count = Counter([v["verb-hypernym"] for v in merged_splits.values()])

    sorted_hyp_keys = sorted(hypernym_count.keys())
    balanced = []
    visited = set()
    min_sum = 0
    for h in sorted_hyp_keys:
        h_count = hypernym_count[h]
        state_inverse = cos_verbs[h]["state-inverse"]
        reverse_h_count = hypernym_count[state_inverse]
        if h in visited:
            continue
        if verbose:
            print(f"{h}: {h_count}\t{state_inverse}: {reverse_h_count}")
        visited.add(state_inverse)
        _min_count = min(h_count, reverse_h_count)
        _min_count = _min_count if _min_count < max_verbs else max_verbs
        min_sum += _min_count
        if _min_count == 0:
            continue
        balanced.append({"verb": h, "reverse": state_inverse, "count": _min_count})
    if verbose:
        print(min_sum)
        for elem in balanced:
            print(elem)
    with open("output/statistics/balanced_verbs.json", "w") as f:
        json.dump(balanced, f)
    return balanced


def sample_balanced(dataset, balanced, seed=42):
    merged_splits = {k: v for d in dataset for k, v in d.items()}
    sampled = {}
    keys = []
    hypernyms = []
    for k, v in merged_splits.items():
        keys.append(k)
        hypernyms.append(v["verb-hypernym"])

    df = pd.DataFrame({"key": keys, "hypernym": hypernyms})
    _c = 0
    for verb_pair in balanced:
        verb = verb_pair["verb"]
        reverse = verb_pair["reverse"]
        count = verb_pair["count"]

        verb_df = df[df["hypernym"] == verb]
        reverse_df = df[df["hypernym"] == reverse]

        sampled_keys = list(verb_df["key"].sample(count, random_state=seed)) + list(
            reverse_df["key"].sample(count, random_state=seed)
        )

        for sampled_k in sampled_keys:
            sampled[_c] = merged_splits[sampled_k]
            _c += 1

        # sampled.update({k: merged_splits[k] for k in sampled_keys})

    with open("output/final/balanced_data.json", "w") as f:
        json.dump(sampled, f)

    print(f"- done! Sampled: {_c} items")


if __name__ == "__main__":
    datasets = ["coin", "ikea", "rareact", "smsm", "star", "yc"]
    loaded = [
        load_processed_dataset(d, level=3, model_size="lg", max_captions=None)
        for d in datasets
    ]

    for d, d_name in zip(loaded, datasets):
        for i, split in enumerate(["train", "val", "test"]):
            for k, v in d[i].items():
                # v.update({"dataset": d_name, "original_split": split})
                v.update({"original_split": split})

    _train = {}
    _val = {}
    _test = {}
    for d in loaded:
        _train.update(d[0])
        _val.update(d[1])
        _test.update(d[2])

    # get_balanced_verb_distribution([_train, _val, _test], load_cos_verbs("triggers/mylist.csv"))

    balanced = balance_dataset(
        [_train, _val, _test], load_cos_verbs("triggers/mylist.csv"), max_verbs=25
    )
    sample_balanced([_train, _val, _test], balanced)
