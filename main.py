"""
- Datasets: COIN, SMSM, YouCook2, RareAct
- change-of-state, prestate, poststate, state-inverse: triggers/updated_concept_cos.csv

TODO: 1. balance out the true_caption/foiled_caption verb distribution [in progress]
TODO: 2. ikea dataset caption parser to be improved
TODO: 3. violin dataset

1. Scan dataset for triggers:
    1.1. datasets have different formats: load the propietary data and convert to a common format
    1.2. prioritize test/devel spltits over train splits 
2. Extract object and subject, time-span
3. Generate foils
4. we could post-process the sentences with ChatGPT to make them more natural?
"""

import os
import json
from argparse import ArgumentParser
from collections import Counter
from time import time

from tqdm import tqdm

from datamanager.dataloader import (
    get_dataset_path,
    load_cos_verbs,
    load_original_dataset,
    save_dataset_splits,
    load_processed_dataset,
    get_verb2cos,
)
from balance_data import cluster_around_cos

from foiler import create_foils
from parsing import phrasal_verb_recognizer, get_object_phrase, init_parser
from exceptions import EXCEPTIONS as manual_exceptions
from exceptions import EXCLUDED as manual_excluded
from exceptions import MANUALLY_EXCLUDED_VERBS

MODEL = "en_core_web_trf"
nlp = init_parser(model=MODEL)


def filter_captions(captions, cos_verbs, add_subject=False, max_captions=None):
    cluster_around_cos()  # build "triggers/verb2cos_mapping.json" from "triggers/mylist.csv"
    verb2cos_mapping = get_verb2cos()
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
        if add_subject:
            parsed = nlp(f'I {v["sentence"]}')
        else:
            parsed = nlp(v["sentence"])
        root = phrasal_verb_recognizer(parsed)
        verbs.append(root)

        if root in MANUALLY_EXCLUDED_VERBS:
            continue

        # manage some edge cases that we have manually checked
        is_exception = False
        if v["sentence"] in manual_exceptions:
            is_exception = True
            _manual = manual_exceptions[v["sentence"]]
            # overwrite the root verb extracted by spacy parser
            root = _manual["verb"]
            my_object = _manual["object"]

        controlled_cos = verb2cos_mapping.get(root, None)

        if (
            controlled_cos in cos_verbs.keys()
        ):  # TODO: cos_verbs list could be expanded with synonyms via wordnet
            v["verb"] = root
            v["verb-hypernym"] = controlled_cos
            v["object"] = get_object_phrase(parsed) if not is_exception else my_object
            v["pre-state"] = cos_verbs[controlled_cos]["pre-state"]
            v["post-state"] = cos_verbs[controlled_cos]["post-state"]
            v["state-inverse"] = cos_verbs[controlled_cos]["state-inverse"]
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
                split, cos_verbs=cos_verbs, max_captions=max_captions, add_subject=False
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

    return to_sample


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


def save_actions_statistics(counter, c_filtered, dataset_name, model_size=None):
    print(f"- Saving dataset action counts")
    fp = os.path.join("output", "statistics", dataset_name)
    os.makedirs(fp, exist_ok=True)
    fp_all = os.path.join(
        fp, f"all_actions_{model_size}.json" if model_size else f"all_actions.json"
    )
    fp_filtered = os.path.join(
        fp,
        f"filtered_actions_{model_size}.json"
        if model_size
        else f"filtered_actions.json",
    )
    with open(fp_all, "w") as f:
        json.dump(dict(counter.most_common()), f)
    with open(fp_filtered, "w") as f:
        json.dump(dict(c_filtered.most_common()), f)
    return


def main(args):
    """
    Pipeline: format -> filter -> foil -> balance
    """
    level = args.load
    model_size = MODEL.split("_")[-1]
    assert 0 <= level <= 3, "Load level (--load 'int') must be in [0, 3]"
    print(
        f"- running pipeline: {['formatting', 'filtering', 'foiling', 'balancing'][level:]}"
    )

    cos_mapping = load_cos_verbs(
        args.cos_verbs, augment_it=args.augment
    )  # TODO: data augmentation should be a pre-processing step to make sure that it makes sense!

    foil_types = ["action", "inverse"]  # TODO: hard-coded

    if level == 0:
        dataset = load_original_dataset(get_dataset_path(args.dataset))
        save_dataset_splits(
            dataset=dataset,
            dataset_name=args.dataset,
            level="formatted",
            model_size=model_size,
        )

    if level <= 1:
        if level == 1:
            dataset = load_processed_dataset(
                dataset_name=args.dataset,
                level=level,
                model_size=model_size,
                max_captions=args.max_captions,
            )

        filtered_dataset, counter_all_actions, counter_filtered = filter_dataset(
            dataset, cos_mapping, max_captions=args.max_captions
        )

        save_dataset_splits(
            dataset=filtered_dataset,
            dataset_name=args.dataset,
            level="filtered",
            model_size=model_size,
            max_captions=args.max_captions,
        )

        save_actions_statistics(
            counter_all_actions,
            counter_filtered,
            args.dataset,
            model_size=model_size,
        )

    if level <= 2:
        if level == 2:
            filtered_dataset = load_processed_dataset(
                dataset_name=args.dataset,
                level=level,
                model_size=model_size,
                max_captions=args.max_captions,
            )
        print(f"- Foiling types: {foil_types}")
        for i in range(len(filtered_dataset)):
            for k, v in filtered_dataset[i].items():
                filtered_dataset[i][k].update(create_foils(v, foil_types=foil_types))

        save_dataset_splits(
            dataset=filtered_dataset,
            dataset_name=args.dataset,
            level="foiled",
            model_size=model_size,
            max_captions=args.max_captions,
        )

    exit("- [DEVEL] Skipping balancing step!")
    if level <= 3:
        # TODO: balancing step should be carried out over all of the merged and filtered together ...
        if level == 3:
            filtered_dataset = load_processed_dataset(
                dataset_name=args.dataset,
                level=level,
                model_size=model_size,
                max_captions=args.max_captions,
            )
        balance_dataset(filtered_dataset, cos_mapping)

    exit()


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
        default="triggers/mylist.csv",
    )
    argparser.add_argument("-n", "--max_captions", type=int, default=None)
    argparser.add_argument("-m", "--model", type=str, default="en_core_web_trf")
    argparser.add_argument("--load", type=int, default=0)
    argparser.add_argument("--augment", action="store_true")
    args = argparser.parse_args()
    main(args)
