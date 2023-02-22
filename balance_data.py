import json
from collections import Counter, defaultdict

import spacy
from nltk.corpus import wordnet as wn

from datamanager.dataloader import load_cos_verbs, load_processed_dataset, load_verblist
from exceptions import MANUALLY_EXCLUDED_VERBS, hardcoded_verb_preprocessing


def cluster_around_cos():
    model = spacy.load("en_core_web_lg")
    cos = load_cos_verbs("triggers/mylist.csv")
    cos_list = list(model.pipe(list(cos.keys())))
    verbs = list(model.pipe(load_verblist()))
    word2cos = {}
    for verb in [
        v
        for v in verbs
        if v[0].pos_ == "VERB" and str(v) not in MANUALLY_EXCLUDED_VERBS
    ]:
        _verb = hardcoded_verb_preprocessing(str(verb))
        top_cos = get_top_similar(model, _verb, cos_list, 1, threshold=0.3)
        word2cos[str(verb)] = top_cos[0] if len(top_cos) > 0 else str(verb)
    print("ok")
    with open("triggers/verb2cos_mapping.json", "w") as f:
        json.dump(word2cos, f)
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
    # TODO: phrasal verb similarity
    _word = model(word)
    if check_pos:
        sims = [
            (str(word), str(c), _word.similarity(c))
            for c in candidates
            if c[0] != _word[0] and c[0].pos_ == "VERB"
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


if __name__ == "__main__":
    cluster_around_cos()
