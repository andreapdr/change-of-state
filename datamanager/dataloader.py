import json
import os
from itertools import chain

import pandas as pd
from nltk.corpus import wordnet as wn


def get_dataset_path(datasetname):
    datasetname = datasetname.lower()
    if datasetname == "coin":
        return os.path.expanduser("~/datasets/COIN/annotations/COIN.json")
    elif datasetname == "smsm":
        return os.path.expanduser("~/datasets/something-something/")
    elif datasetname == "yc":
        return os.path.expanduser("~/datasets/youcook2/fixed_youcook2_trainval.json")
    elif datasetname == "rareact":
        return os.path.expanduser("~/datasets/rareAct/rareact.csv")
    elif datasetname == "ikea":
        return os.path.expanduser("~/datasets/ikea_asm/gt_segments.json")
    elif datasetname == "violin":
        return os.path.expanduser("~/datasets/violin/violin_annotation.json")
    else:
        raise ValueError(f"Dataset {datasetname} not supported!")


def load_original_dataset(path):
    path = os.path.expanduser(path)
    if "coin" in path.lower():
        dataset = load_coin(path)
    elif "something-something" in path.lower():
        dataset = load_smsm(path)
    elif "youcook" in path.lower():
        dataset = load_yc(path)
    elif "rareact" in path.lower():
        dataset = load_rareact(path)
    elif "ikea" in path.lower():
        dataset = load_ikea(path)
    elif "violin" in path.lower():
        dataset = load_violin(path)
    else:
        raise ValueError(f"Dataset not supported!")
    print(
        f"- Len train: {len(dataset[0])} - len val: {len(dataset[1])} - len test: {len(dataset[2])}"
    )
    return dataset


def load_violin(path):
    dataset = json.load(open(path))
    train, val, test = {}, {}, {}

    for k, v in dataset.items():
        if v["split"] == "train":
            train[k] = v
        elif v["split"] == "val":
            val[k] = v
        elif v["split"] == "test":
            test[k] = v
        else:
            raise ValueError(f"Split {v['split']} not supported!")
    return (train, val, test)


def load_ikea(path):
    dataset = json.load(open(path))["database"]
    train, val, test = {}, {}, {}

    for k, v in dataset.items():
        unfolded = _unfold_ikea_entry(k, v)
        if v["subset"] == "training":
            for sub_k, sub_elem in unfolded.items():
                train[sub_k] = sub_elem
        elif v["subset"] == "validation":
            for sub_k, sub_elem in unfolded.items():
                val[sub_k] = sub_elem
        elif v["subset"] == "testing":
            for sub_k, sub_elem in unfolded.items():
                test[sub_k] = sub_elem
        else:
            raise ValueError(f"Subset {v['subset']} not supported!")

    return (train, val, test)


def _unfold_ikea_entry(key, entry):
    unfolded = {}
    for i, annotation in enumerate(entry["annotation"]):
        new_key = f"{key}_{i}"
        unfolded[new_key] = {
            "top_level_key": key,
            "annotation_id": new_key,
            "sentence": annotation["label"],
            "timestamp": annotation["segment"],
            "video_id": key,
            "split": entry["subset"],
        }
    return unfolded


def load_rareact(path):
    """
    filter out sentences with annotation == 1.
    From https://github.com/antoine77340/RareAct
    Annotation for the given clip and (verb, noun) class.
    1: Positive.
    2: Hard negative (only verb is right):
    3: Hard negative (only noun is right).
    4: Hard negative (Both verb and noun are valid but verb is not applied to noun).
    0: Negative.
    # TODO: train/val/test split ???
    """
    df = pd.read_csv(path)
    df = df.loc[df["annotation"] == 1]

    train, val, test = {}, {}, {}

    # convert df to dict of dict
    for i, row in df.iterrows():
        train[row["id"]] = {
            "sentence": _create_sentence_rareact(row.verb, row.noun),
            "timestamp": [row.start, row.end],
            "video_id": row.video_id,
            "split": "training",
        }
    return (train, val, test)


def _create_sentence_rareact(verb, noun):
    return f"{verb} the {noun}"


def load_coin(path):
    dataset = json.load(open(path))["database"]
    train, val, test = {}, {}, {}
    for k, v in dataset.items():
        unfolded = _unfold_coin_entry(k, v)
        if v["subset"] == "training":
            for sub_k, sub_elem in unfolded.items():
                train[sub_k] = sub_elem
        elif v["subset"] == "validation":
            for sub_k, sub_elem in unfolded.items():
                val[sub_k] = sub_elem
        elif v["subset"] == "testing":
            for sub_k, sub_elem in unfolded.items():
                test[sub_k] = sub_elem
        else:
            raise ValueError(f"Unknown subset: {v['subset']}")
    return (train, val, test)


def _unfold_coin_entry(key, entry):
    unfolded = {}
    for annotation in entry["annotation"]:
        new_key = f"{key}_{annotation['id']}"
        unfolded[new_key] = {
            "top_level_key": key,
            "annotation_id": new_key,
            "sentence": annotation["label"],
            "timestamp": annotation["segment"],
            "video_id": entry["video_url"],
            "split": entry["subset"],
        }
    return unfolded


def load_yc(path):
    """
    Each item in the dataset corresponds to a video.
    A video has a list of captions in the "annotations" field.
    Each annotations is denoted as a "segment", where each segment
    has a "segment" field, which is a list of [start, end] timestamps,
    an "id" and the "sentence" field containing the caption itself.
    We unfold all of the segments into a list of captions.
    The new ids will be the old top-level key + the segment id.

    TODO: we also have some annotations like bounding boxes of objects present in the scene!
    """
    dataset = json.load(open(path))["database"]

    train, val, test = {}, {}, {}

    for k, v in dataset.items():
        unfolded = _unfold_yc_entry(k, v)
        if v["subset"] == "training":
            for sub_k, sub_elem in unfolded.items():
                train[sub_k] = sub_elem
        elif v["subset"] == "validation":
            for sub_k, sub_elem in unfolded.items():
                val[sub_k] = sub_elem
        elif v["subset"] == "testing":
            for sub_k, sub_elem in unfolded.items():
                test[sub_k] = sub_elem
        else:
            raise ValueError(f"Unknown subset: {v['subset']}")
    return (train, val, test)


def load_smsm(path):
    """
    Path for smsm should be the folder path. We then take care of loading
    the file containing ids for the splits (train, validation, test.json) and the
    file containing the unique (?) captions (labels.json).

    TODO: We can filter out appropriate sentences from the labels file!
    """

    path = os.path.join(os.path.expanduser(path), "labels")
    train = json.load(open(os.path.join(path, "train.json")))
    val = json.load(open(os.path.join(path, "validation.json")))
    test = json.load(
        open(os.path.join(path, "test.json"))
    )  # test set in hidden (unlabeled)

    all_captions = json.load(open(os.path.join(path, "labels.json")))

    # convert list of dict to dict of dict
    train = {k["id"]: _unfold_smsm_entry(k["id"], k, split="train") for k in train}
    val = {k["id"]: _unfold_smsm_entry(k["id"], k, split="val") for k in val}
    # test = {k["id"]: _unfold_smsm_entry(k["id"], k) for k in test}
    test = {}

    return (train, val, test)  # TODO: we are discarding the labels file here!


def _unfold_smsm_entry(key, entry, split):
    """
    Unfold a single entry from SMSM dataset.
    """
    new_entry = {
        "top_level_key": key,
        "annotation_id": entry["id"],
        "sentence": entry["label"],
        "timestamp": [None, None],
        "video_id": entry["id"],
        "split": split,
    }
    return new_entry


def _unfold_yc_entry(key, entry):
    """
    Unfold a single entry from YouCook2 dataset.
    """
    new_entry = {}
    for segment in entry["annotations"]:
        new_entry[f'{key}_{segment["id"]}'] = {
            "top_level_key": key,
            "annotation_id": segment["id"],
            "sentence": segment["sentence"],
            "timestamp": segment["segment"],
            "video_id": entry["video_url"],
            "split": entry["subset"],
        }
    return new_entry


def load_cos_verbs(path, agument_it=False):
    """
    Load change-of-state verbs from csv file.
    Each row consits of verb, (up to 3) prestate,
    (up to 3) poststate, (up to 3) state-inverse.
    Returns a dict in the form of
    {
        cos_verb: {
            prestate: [list of prestates],
            poststate: [list of poststates],
            state-inverse: [list of state-inverses]
        }
    }
    """

    path = os.path.expanduser(path)
    cos_df = pd.read_csv(path)

    cos_dict = {}

    # change of state verbs
    cos_verbs = cos_df.cos.to_list()

    # prestates
    prestate = cos_df.pre.to_list()
    # prestate2 = cos_df.pre2.to_list()
    # prestate3 = cos_df.pre3.to_list()

    # poststates
    poststate = cos_df.post.to_list()
    # poststate2 = cos_df.post2.to_list()
    # poststate3 = cos_df.post3.to_list()

    # inverse states
    inverse = cos_df.inv.to_list()
    # inverse2 = cos_df.inv2.to_list()
    # inverse3 = cos_df.inv3.to_list()

    # create dict
    for (
        v,
        pre,
        post,
        inv,
    ) in zip(
        cos_verbs,
        prestate,
        poststate,
        inverse,
    ):
        cos_dict[v] = {
            "pre-state": pre,
            "post-state": post,
            "state-inverse": inv,
        }

    if agument_it:
        cos_dict = augment_cos(cos_dict)

    return cos_dict


def augment_cos(cos_dict, max_syns=3):
    print(
        f"- Augmenting change-of-state verbs via WordNet synonyms (up to {max_syns})..."
    )
    new_dict = {}
    for k, v in cos_dict.items():
        syns = get_synonyms(k, max_synonyms=max_syns)
        new_dict[k] = v
        for syn in syns:
            new_dict[syn] = v
    return new_dict


def get_synonyms(verb, max_synonyms=10):
    # TODO: not the most appropriate placement for this function
    syns = set()
    syns = wn.synsets(verb, pos=wn.VERB)
    lemmas = sum([word.lemma_names() for word in syns], [])
    synonyms = [w.replace("_", " ") for w in lemmas if w.replace("_", " ") != verb]
    return synonyms[:max_synonyms]


def save_dataset_splits(dataset, dataset_name, level, max_captions):
    assert level in ["formatted", "processed", "filtered"]
    for split, data in zip(["train", "val", "test"], dataset):
        save_dataset(
            dataset=data,
            path=os.path.join("output", level, dataset_name),
            fname=dataset_name,
            split=split,
            nrows=max_captions
        )


def save_dataset(dataset, path, fname, split, nrows=None):
    """
    Save dataset to disk.
    """
    path = os.path.expanduser(path)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, fname)
    filepath = (
        f"{path}_{split}.json" if nrows is None else f"{path}_{split}_{nrows}.json"
    )
    print(f"- Saving dataset to {filepath}")
    with open(filepath, "w") as f:
        json.dump(dataset, f)


def load_processed_dataset(dataset_name, max_captions):
    fdir = os.path.join("output", dataset_name)
    dataset = []
    for split in ["train", "val", "test"]:
        filepath = (
            f"{dataset_name}_{split}.json"
            if max_captions is None
            else f"{dataset_name}_{split}_{max_captions}.json"
        )
        with open(os.path.join(fdir, filepath)) as f:
            dataset.append(json.load(f))
    return dataset
