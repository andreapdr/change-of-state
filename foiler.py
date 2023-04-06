from lemminflect import getInflection

from manual_exceptions.intransitive_verbs import INTRANSITIVE

# TODO: manage missing articles before nouns


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
    particle = None
    if " " in verb:
        verb, particle = verb.split(" ")
    infl = getInflection(verb, tag=tag)
    if len(infl) == 0:
        print(f"- Error inflecting '{verb}' with tag '{tag}'")
        infl = ["[UNK]"]
    return infl[0], particle


def not_transitive(verb):
    if verb in INTRANSITIVE:
        return True
    else:
        return False


def is_plural(subject):
    if subject[-1] == "s" or "and" in subject:
        if subject[-2] == "s":  # e.g., "glass"
            return False
        return True
    elif " and " in subject:
        return True
    else:
        return False


def manage_aux(subject):
    if subject is None:
        raise ValueError("Subject cannot be None")
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
    _object = sentence["object"] if sentence["object"] else "something"
    tag = "VBZ" if not is_plural(_object) else "VBP"
    _verb, _particle = inflect(sentence["verb"], tag=tag)
    _verb_passive, _particle_passive = inflect(sentence["verb"], tag="VBZ")
    _inverse, _particle_inv = inflect(sentence["state-inverse"], tag=tag)
    _inverse_passive, _particle_inv_passive = inflect(
        sentence["state-inverse"], tag="VBZ"
    )

    if not_transitive(sentence["verb"]):
        capt = f"{_object.capitalize()} {_verb}{' ' + _particle if _particle else ''}."
        foil = f"{_object.capitalize()} {_inverse}{' ' + _particle_inv if _particle_inv else ''}."
    else:
        capt = f"Someone {_verb_passive} {_object}{' ' + _particle_passive if _particle_passive else ''}."
        foil = f"Someone {_inverse_passive} {_object}{' ' +_particle_inv_passive if _particle_inv_passive else ''}."
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
    _verb, _particle = inflect(sentence["verb"], tag="VBZ")
    _inverse, _particle_inv = inflect(sentence["state-inverse"], tag="VBZ")
    _object = sentence["object"] if sentence["object"] else "something"
    aux = manage_aux(_object)

    if not_transitive(sentence["verb"]):
        capt = f"Initially, {_object} {aux} {sentence['pre-state']}. Then, {_object} {_verb}{' ' + _particle if _particle else ''}. At the end, {_object} {aux} {sentence['post-state']}."
        foil = f"Initially, {_object} {aux} {sentence['post-state']}. Then, {_object} {_inverse}{' ' + _particle_inv if _particle_inv else ''}. At the end, {_object} {aux} {sentence['pre-state']}."
    else:
        capt = f"Initially, {_object} {aux} {sentence['pre-state']}. Then, someone {_verb} {_object}{' ' + _particle if _particle else ''}. At the end, {_object} {aux} {sentence['post-state']}."
        foil = f"Initially, {_object} {aux} {sentence['post-state']}. Then, someone {_inverse} {_object}{' ' + _particle_inv if _particle_inv else ''}. At the end, {_object} {aux} {sentence['pre-state']}."
    return capt, foil


if __name__ == "__main__":
    import json
    from os.path import expanduser

    foil_types = ["action", "pre-state", "post-state", "inverse"]

    datapath = expanduser("~/datasets/vl-bench/change-state.json")
    outpath = expanduser("~/datasets/vl-bench/newfoil.json")

    data = json.load(open(datapath))
    print(f"- foiling data at {datapath} - {len(data)} examples")

    foiled_dataset = {}
    for k, v in data.items():
        _foils = create_foils(v["change_of_state"], foil_types=foil_types)
        foiled_dataset[k] = v
        foiled_dataset[k].update(_foils)

    json.dump(foiled_dataset, open(outpath, "w"))
