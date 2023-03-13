"""
- Datasets: COIN, SMSM, YouCook2, RareAct
- change-of-state, prestate, poststate, state-inverse: triggers/updated_concept_cos.csv

TODO: 1. balance out the true_caption/foiled_caption verb distribution [in progress]
TODO: 2. ikea dataset caption parser to be improved

1. Scan dataset for triggers:
    1.1. datasets have different formats: load the propietary data and convert to a common format
    1.2. prioritize test/devel spltits over train splits 
2. Extract object and subject, time-span
3. Generate foils
4. we could post-process the sentences with ChatGPT to make them more natural?
"""

import json
import os
from argparse import ArgumentParser
from collections import Counter
from time import time

from tqdm import tqdm

from datamanager.dataloader import (
    get_dataset_path,
    get_verb2cos,
    load_cos_verbs,
    load_original_dataset,
    load_processed_dataset,
    save_dataset_splits,
)
from foiler import create_foils
from manual_exceptions.exceptions import EXCEPTIONS as manual_exceptions
from manual_exceptions.excluded_sents import EXCLUDED as manual_excluded
from manual_exceptions.excluded_verbs import MANUALLY_EXCLUDED_VERBS
from parsing import get_object_phrase, init_parser, phrasal_verb_recognizer

MODEL = "en_core_web_lg"
# MODEL = "en_core_web_trf"
nlp = init_parser(model=MODEL)


def filter_captions(captions, cos_verbs, add_subject=False, max_captions=None):
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

        if root == "put":
            root = "put in"
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
            has_obj = v.get("object", None)
            if has_obj is None:
                v["object"] = (
                    get_object_phrase(parsed) if not is_exception else my_object
                )
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
    hypernym_count = Counter([v["verb-hypernym"] for v in merged_splits.values()])

    sorted_hyp_keys = sorted(hypernym_count.keys())
    visited = set()
    for h in sorted_hyp_keys:
        h_count = hypernym_count[h]
        state_inverse = cos_verbs[h]["state-inverse"]
        if h in visited:
            continue
        print(f"{h}: {h_count}\t{state_inverse}: {hypernym_count[state_inverse]}")
        visited.add(state_inverse)


def save_actions_statistics(
    counter, c_filtered, dataset_name, model_size=None, max_captions=None
):
    print(f"- Saving dataset action counts")
    fp = os.path.join("output", "statistics", dataset_name)
    os.makedirs(fp, exist_ok=True)

    filename = f"all_actions_{model_size}" if model_size else f"all_actions"
    filename = (
        f"{filename}.json"
        if max_captions is None
        else f"{filename}_{max_captions}.json"
    )

    filenamefiltered = (
        f"filtered_actions_{model_size}" if model_size else f"filtered_actions"
    )
    filenamefiltered = (
        f"{filenamefiltered}.json"
        if max_captions is None
        else f"{filenamefiltered}_{max_captions}.json"
    )

    fp_all = os.path.join(fp, filename)
    fp_filtered = os.path.join(fp, filenamefiltered)

    with open(fp_all, "w") as f:
        print(f"- saving dataset action statistics to: {fp_all}")
        json.dump(dict(counter.most_common()), f)
    with open(fp_filtered, "w") as f:
        print(f"- saving filtrered dataset action statistics to: {fp_filtered}")
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

    cos_mapping = load_cos_verbs(args.cos_verbs, augment_it=args.augment)

    foil_types = ["action", "pre-state", "post-state", "inverse"]  # TODO: hard-coded

    if level == 0:
        dataset = load_original_dataset(get_dataset_path(args.dataset))
        save_dataset_splits(
            dataset=dataset,
            dataset_name=args.dataset,
            level="formatted",
        )

    if level <= 1:
        if level == 1:
            dataset = load_processed_dataset(
                dataset_name=args.dataset,
                level=level,
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
            max_captions=args.max_captions,
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

    if level <= 3:
        exit("Balancing should be carried out on all the datasets together")
        # TODO: balancing step should be carried out over all of the merged and filtered together ...
        if level == 3:
            filtered_dataset = load_processed_dataset(
                dataset_name=args.dataset,
                level=level,
                model_size=model_size,
                max_captions=args.max_captions,
            )

        # balance_dataset(filtered_dataset, cos_mapping)
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
