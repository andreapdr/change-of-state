import json
import spacy

from datamanager.dataloader import load_cos_verbs, load_processed_dataset, load_verblist
from manual_exceptions.manual_verb2cos import hardcoded_verb_preprocessing
from manual_exceptions.excluded_verbs import MANUALLY_EXCLUDED_VERBS


def get_action2cos_mapping(
    model, max_captions=None
):  # TODO: move this to another module
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


def get_top_similar(model, word, candidates, k, check_pos=True, threshold=0.5):
    _word = model(word)
    if check_pos:
        sims = [
            (str(word), str(c), _word.similarity(c))
            for c in candidates
            if (c[0] != _word[0] and c[0].pos_ == "VERB")
            or c[0].orth_ in ["close", "unwrap", "break", "empty"]
        ]
    else:
        sims = [(str(word), str(c), _word.similarity(c[0])) for c in candidates]
    sims.sort(key=lambda x: x[2], reverse=True)
    sims = [s[1] for s in sims if s[2] >= threshold]
    return sims[:k]


if __name__ == "__main__":
    get_action2cos_mapping(model="en_core_web_lg", max_captions=None)
