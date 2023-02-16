import spacy


def init_parser(model="en_core_web_trf"):
    print(f"- Loading spacy model {model}")
    # nlp = spacy.load("en_core_web_sm")
    return spacy.load(model)


def phrasal_verb_recognizer(parsed):
    root = None
    for token in parsed:
        if token.dep_ == "ROOT":
            root = token.orth_
        if (
            token.dep_ == "prt"
            and token.head.pos_ == "VERB"  # TODO: this is probably useless?
            and token.head.dep_ == "ROOT"
        ):
            verb = token.head.orth_
            particle = token.orth_
            return f"{verb} {particle}"
    return root


def get_object_phrase(parsed):
    for token in parsed:
        if "dobj" in token.dep_:
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return parsed[start:end].orth_


def get_dative_phrase(parsed):
    for token in parsed:
        if "dative" in token.dep_:
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return parsed[start:end].orth_
