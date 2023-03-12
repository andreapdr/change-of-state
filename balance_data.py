import json
from collections import Counter, defaultdict

import spacy
from nltk.corpus import wordnet as wn

from datamanager.dataloader import load_cos_verbs, load_processed_dataset, load_verblist

from manual_exceptions.forced_allowed_verbs import MANUALLY_ALLOWED_VERBS
from manual_exceptions.excluded_verbs import MANUALLY_EXCLUDED_VERBS
from manual_exceptions.manual_verb2cos import hardcoded_verb_preprocessing


def cluster_around_cos(model, max_captions=None):
    model_size = model.split("_")[-1]
    model = spacy.load(model)
    cos = load_cos_verbs("triggers/mylist.csv")
    cos_list = list(model.pipe(list(cos.keys())))
    verbs = list(model.pipe(load_verblist(model_size=model_size, n=max_captions)))
    word2cos = {}
    for verb in [
        v
        for v in verbs
        if v[0].pos_ == "VERB" and str(v) not in MANUALLY_EXCLUDED_VERBS
    ]:
        _verb = hardcoded_verb_preprocessing(str(verb))
        top_cos = get_top_similar(model, _verb, cos_list, 1, threshold=0.3)
        word2cos[str(verb)] = top_cos[0] if len(top_cos) > 0 else str(verb)
    with open("triggers/verb2cos_mapping.json", "w") as f:
        json.dump(dict(sorted(word2cos.items())), f)
    return


def postprocess_via_wn(verb, verb_constraint):
    # TODO: wordnet does not seem to manage well pharasal verbs
    verb = str(verb).split(" ")[0]
    verb_synonyms = _get_wn_synonyms(verb)
    verb_antonyms = _get_wn_antonyms(verb)
    if verb_constraint is not None:
        verb_synonyms = [s for s in verb_synonyms if s in verb_constraint]
        verb_antonyms = [s for s in verb_antonyms if s in verb_constraint]
    return verb_synonyms, verb_antonyms


def get_top_similar(model, word, candidates, k, check_pos=True, threshold=0.5):
    _word = model(word)
    if check_pos:
        sims = [
            (str(word), str(c), _word.similarity(c))
            # for c in candidates
            # if c[0].pos_ == "VERB" or c[0].orth_ in ["close", "unwrap", "break", "empty"]
            for c in candidates
            if (c[0] != _word[0] and c[0].pos_ == "VERB")
            or c[0].orth_ in ["close", "unwrap", "break", "empty"]
        ]
    else:
        sims = [(str(word), str(c), _word.similarity(c[0])) for c in candidates]
    sims.sort(key=lambda x: x[2], reverse=True)
    sims = [s[1] for s in sims if s[2] >= threshold]
    return sims[:k]


def _get_wn_antonyms(word):
    antonyms = []
    for syn in wn.synsets(word, pos=wn.VERB):
        for l in syn.lemmas():
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    return sorted([a.replace("_", " ") for a in set(antonyms)])


def _get_wn_synonyms(word):
    synonyms = [
        sorted(list(set(ss.lemma_names(lang="eng")) - {word}))
        for ss in wn.synsets(word, lang="eng", pos=wn.VERB)
    ]
    synonyms = sorted(list(set(sum(synonyms, []))))
    return [s.replace("_", " ") for s in synonyms]


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


def old_balance_dataset(dataset, cos_verbs, delta=0.1):
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


if __name__ == "__main__":
    # cluster_around_cos(model="en_core_web_lg", max_captions=None)
    # exit()
    datasets = ["coin", "ikea", "rareact", "smsm", "star", "yc"]
    loaded = [
        load_processed_dataset(d, level=3, model_size="lg", max_captions=None)
        for d in datasets
    ]
    _train = {}
    _val = {}
    _test = {}
    for d in loaded:
        _train.update(d[0])
        _val.update(d[1])
        _test.update(d[2])

    # balance_dataset([_train, _val, _test], load_cos_verbs("triggers/mylist.csv"))
    old_balance_dataset([_train, _val, _test], load_cos_verbs("triggers/mylist.csv"))
