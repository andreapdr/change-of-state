from lemminflect import getInflection
from exceptions import INTRANSITIVE


PTB_TAG = {
    "VB": "base form",
    "VBD": "past tense",
    "VBG": "gerund or presnet participle",
    "VBN": "past participle",
    "VBP": "non-3rd person singular present",
    "VBZ": "3rd person singular present",
}


def inflect(verb, tag):
    assert tag in PTB_TAG, f"pos '{tag}' not in {list(tag.keys())}"
    # simple manager of multi-word verbs - maybe append the second part of the verb?
    if " " in verb:
        verb = verb.split(" ")[0]
    infl = getInflection(verb, tag=tag)
    if len(infl) == 0:
        print(f"- Error inflecting '{verb}' with tag '{tag}'")
        infl = ["[UNK]"]
    return infl[0]


def not_transitive(verb):
    if verb in INTRANSITIVE:
        return True
    else:
        return False


def is_plural(subject):
    if subject[-1] == "s" or "and" in subject:
        return True
    else:
        return False


def manage_aux(subject):
    if is_plural(subject):
        return "are"
    else:
        return "is"


def create_foils(sentence, foil_types=["action"]):
    foils = {}
    switcher = {
        "action": get_action_capt_and_foil,
        "pre-state": get_prestate_capt_and_foil,
        "post-state": get_poststate_capt_and_foil,
        "inverse": get_inverse_capt_and_foil,
    }
    for ft in foil_types:
        foils[ft] = switcher[ft](sentence)

    return {"foils": foils}


def get_action_capt_and_foil(sentence):
    tag = "VBZ" if not is_plural(sentence["object"]) else "VBP"
    _verb = inflect(sentence["verb"], tag=tag)
    _verb_3rd = inflect(sentence["verb"], tag="VBZ")
    _inverse_3rd = inflect(sentence["state-inverse"], tag="VBZ")
    _inverse = inflect(sentence["state-inverse"], tag=tag)
    _object = sentence["object"] if sentence["object"] else "Someone"

    if not_transitive(sentence["verb"]):
        capt = f"{_object.capitalize()} {_verb}."
        foil = f"{_object.capitalize()} {_inverse}"
    else:
        capt = f"Someone {_verb_3rd} {_object}."
        foil = f"Someone {_inverse_3rd} {_object}."
    return capt, foil


def get_prestate_capt_and_foil(sentence):
    _object = sentence["object"] if sentence["object"] else "Someone"
    aux = manage_aux(_object)

    if not_transitive(sentence["verb"]):
        capt = f"Initially, {_object} {aux} {sentence['pre-state']}."
        foil = f"Initially, {_object} {aux} {sentence['post-state']}."
    else:
        capt = f"Initially, {_object} {aux} {sentence['pre-state']}."
        foil = f"Initially, {_object} {aux} {sentence['post-state']}."
    return capt, foil


def get_poststate_capt_and_foil(sentence):
    _object = sentence["object"] if sentence["object"] else "Someone"
    aux = manage_aux(_object)

    if not_transitive(sentence["verb"]):
        capt = f"At the end, {_object} {aux} {sentence['post-state']}."
        foil = f"At the end, {_object} {aux} {sentence['pre-state']}."
    else:
        capt = f"At the end, {_object} {aux} {sentence['post-state']}."
        foil = f"At the end, {_object} {aux} {sentence['pre-state']}."
    return capt, foil


def get_inverse_capt_and_foil(sentence):
    _verb = inflect(sentence["verb"], tag="VBZ")
    _inverse = inflect(sentence["state-inverse"], tag="VBZ")
    _object = sentence["object"] if sentence["object"] else "Someone"
    aux = manage_aux(_object)

    if not_transitive(sentence["verb"]):
        capt = f"Initially, {_object} {aux} {sentence['pre-state']}. Then, {_object} {_verb}. At the end, {_object} {aux} {sentence['post-state']}."
        foil = f"Initially, {_object} {aux} {sentence['post-state']}. Then, {_object} {_inverse}. At the end, {_object} {aux} {sentence['pre-state']}."
    else:
        capt = f"Initially, {_object} {aux} {sentence['pre-state']}. Then, Someone {_verb} {_object}. At the end, {_object} {aux} {sentence['post-state']}."
        foil = f"Initially, {_object} {aux} {sentence['post-state']}. Then, someone {_inverse} {_object}. At the end, {_object} {aux} {sentence['pre-state']}."
    return capt, foil
