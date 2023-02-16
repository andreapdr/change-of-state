"""
- Datasets: COIN, SMSM, YouCook2, RareAct
- change-of-state, prestate, poststate, state-inverse: triggers/updated_concept_cos.csv

TODO: 1. balance out the true_caption/foiled_caption verb distribution

1. Scan dataset for triggers:
    1.1. datasets have different formats: load the propietary data and convert to a common format
    1.2. prioritize test/devel spltits over train splits 
2. Extract object and subject, time-span (???)
3. Generate foils
4. we could post-process the sentences with ChatGPT to make them more natural?
"""

import os
from argparse import ArgumentParser
from collections import Counter
from time import time

from tqdm import tqdm

from datamanager.dataloader import (
    get_dataset_path,
    load_cos_verbs,
    load_original_dataset,
    save_dataset,
    load_processed_dataset,
)

from foiler import create_foils
from parsing import phrasal_verb_recognizer, get_object_phrase, init_parser
from exceptions import EXCEPTIONS as manual_exceptions
from exceptions import EXCLUDED as manual_excluded
from nltk.corpus import wordnet as wn

nlp = init_parser(model="en_core_web_sm")


def filter_captions(captions, cos_verbs, max_captions=None):
    tinit = time()
    verbs = []
    filtered_dataset = {}
    i = 0
    for k, v in tqdm(captions.items()):
        if v["sentence"] in manual_excluded:
            continue
        if i == max_captions:
            break
        # TODO: sometimes could be useful to add the subject "I" (e.g. in COIN dataset for sure)
        # parsed = nlp(f'I {v["sentence"]}')
        parsed = nlp(v["sentence"])
        root = phrasal_verb_recognizer(parsed)
        verbs.append(root)

        # manage some edge cases that we have manually checked
        # TODO: we could move this except manager before the spacy parser to avoid some useless computation
        is_exception = False
        if v["sentence"] in manual_exceptions:
            is_exception = True
            _manual = manual_exceptions[v["sentence"]]
            # overwrite the root verb extracted by spacy parser
            root = _manual["verb"]
            my_object = _manual["object"]

        if root in cos_verbs:
            v["verb"] = root
            v["object"] = get_object_phrase(parsed) if not is_exception else my_object
            v["pre-state"] = cos_verbs[root]["pre-state"]
            v["post-state"] = cos_verbs[root]["post-state"]
            v["state-inverse"] = cos_verbs[root]["state-inverse"]
            filtered_dataset[k] = v
            i += 1
    print(f"Time to filter: {(time() - tinit):.2f}s")
    c = Counter(verbs)
    filtered = [v for v in verbs if v in cos_verbs]
    c_filtered = Counter(filtered)
    return filtered_dataset, c, c_filtered


def filter_dataset(dataset, cos_verbs, max_captions=None):
    """
    Filter dataset to contain only captions with change-of-state verbs.
    """
    filtered_dataset = []
    for split, str_split in zip(dataset, ["train", "val", "test"]):
        print(f"- Filtering {str_split}...")
        if len(split) != 0:
            _filtered_dataset, counter, c_filtered = filter_captions(
                split, cos_verbs=cos_verbs, max_captions=max_captions
            )
        else:
            _filtered_dataset = {}
        filtered_dataset.append(_filtered_dataset)
    print(
        f"- Len filtered train: {len(filtered_dataset[0])} - val: {len(filtered_dataset[1])} - test: {len(filtered_dataset[2])}"
    )
    return filtered_dataset, counter, c_filtered


def balance_dataset(dataset, cos_verbs, delta=0.1):
    """
    Balance dataset to contain the approximately (delta) same number distribution of cos verbs in the true_caption and foiled_caption.
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

    # all_verbs = sum(
    #     [list(_reverse2action_mapping.keys()), list(_action2reverse_mapping.keys())], []
    # )

    action_mapping = {}
    action_mapping.update(_reverse2action_mapping)
    action_mapping.update(_action2reverse_mapping)

    to_sample = _to_sample(
        action_mapping, counter_actual_verbs, counter_reversed_action
    )

    # TODO: remove this print
    for k, v in to_sample.items():
        if v >= 0:
            print(f"{k}: {v}")


def get_synonyms(verb, max_synonyms=10):
    synonyms = set()
    for syn in wn.synsets(verb, pos=wn.VERB)[:max_synonyms]:
        for l in syn.lemmas():
            synonyms.add(l.name().replace("_", " "))
    return list(synonyms)


def _to_sample(action_mapping, counter_actual_verbs, counter_reverse_action):
    # TODO: this logic is a mess
    to_sample = {}
    for verb in action_mapping.keys():
        reverse_verb = action_mapping[
            verb
        ]  # TODO: e.g., to dirty -> to wash but also to wash -> to clean
        actual_count = counter_actual_verbs[verb]
        target_count = counter_reverse_action.get(reverse_verb, 0)
        n_samples = min(actual_count, target_count)
        to_sample[verb] = n_samples
        to_sample[reverse_verb] = n_samples
    return to_sample


def main(args):
    cos_mapping = load_cos_verbs(args.cos_verbs)

    if not args.load:
        dataset = load_original_dataset(get_dataset_path(args.dataset))

        filtered_dataset, _, _ = filter_dataset(
            dataset, cos_mapping, max_captions=args.max_captions
        )

        for i in range(len(filtered_dataset)):
            for k, v in filtered_dataset[i].items():
                filtered_dataset[i][k].update(
                    create_foils(
                        v, foil_types=["action", "pre-state", "post-state", "inverse"]
                    )
                )

        for split, data in zip(["train", "val", "test"], filtered_dataset):
            save_dataset(
                dataset=data,
                path=os.path.join("output", args.dataset),
                split=split,
                nrows=args.max_captions,
            )

    if args.load:
        filtered_dataset = load_processed_dataset(
            dataset_name=args.dataset, max_captions=args.max_captions
        )

    balance_dataset(filtered_dataset, cos_mapping)

    exit(0)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="coin",
        help="Dataset to process",
    )
    argparser.add_argument(
        "-c",
        "--cos_verbs",
        type=str,
        default="triggers/chatgpt_cos.csv",
    )
    argparser.add_argument("-n", "--max_captions", type=int, default=None)
    argparser.add_argument("-m", "--model", type=str, default="en_core_web_trf")
    argparser.add_argument("--load", action="store_true")
    args = argparser.parse_args()
    main(args)
