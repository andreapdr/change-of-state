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
)

from foiler import create_foils
from parsing import phrasal_verb_recognizer, get_object_phrase, init_parser
from exceptions import EXCEPTIONS as manual_exceptions
from exceptions import EXCLUDED as manual_excluded


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
    for split in dataset:
        if len(split) != 0:
            _filtered_dataset, counter, c_filtered = filter_captions(
                split, cos_verbs=cos_verbs, max_captions=max_captions
            )
        else:
            _filtered_dataset = []
        filtered_dataset.append(_filtered_dataset)
    print(
        f"- Len filtered train: {len(filtered_dataset[0])} - val: {len(filtered_dataset[1])} - test: {len(filtered_dataset[2])}"
    )
    return filtered_dataset, counter, c_filtered


def main(args):
    dataset = load_original_dataset(get_dataset_path(args.dataset))
    cos_mapping = load_cos_verbs(args.cos_verbs)

    filtered_dataset, _, _ = filter_dataset(
        dataset, cos_mapping, max_captions=args.max_captions
    )

    for k, v in filtered_dataset[0].items():  # TODO: we are forcing to train set split
        filtered_dataset[0][k].update(
            create_foils(v, foil_types=["action", "pre-state", "post-state", "inverse"])
        )

    for split, data in zip(["train", "val", "test"], filtered_dataset):
        save_dataset(
            dataset=data,
            path=os.path.join("output", args.dataset),
            split=split,
            nrows=args.max_captions,
        )

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
    args = argparser.parse_args()
    main(args)
