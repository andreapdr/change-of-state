def hardcoded_verb_preprocessing(verb):
    verb2cos = {
        "apply": "put on",
        "absorb": "dry",
        "unroll": "unwrap",
        "soak": "wet",
        "boil": "heat up",
        "lift up": "raise",
        "chop up": "cut",
        "break": "disassemble",
        "break down": "disassemble",
        "break apart": "disassemble",
        "foling": "fold",
        "stem": "cut",
        "pierce": "empty",
        "bolt up": "screw",
        "knock": "insert",
        "take out": "pull out",
        "rinse off": "clean",
        "rinse": "wet",
        "switch off": "turn off",
        "turn off": "turn off",
        "connect": "connect",
        "sift": "spread",
        "turn on": "turn on",
        "switch on": "turn on",
        "scoop out": "reveal",
        "start": "turn on",
        "dice up": "disassemble",
        "mash": "disassemble",
        "turn out": "uncover",
        "nail down": "fix",
        "whisk": "comb",
        "fasten": "attach",
        "spread out": "disassemble",
        "cover up": "cover",
        "raise up": "raise",
        "move up": "raise",
        "cut up": "cut",
        "let down": "drop",
        "move down": "drop",
        "stir over": "combine",
        "boil": "heat up",
        "screw down": "screw",
        "place down": "drop",
        "buckle": "attach",
        "spread": "disassemble",
        "dig": "cover",
        "shred": "disassemble",
        "drain off": "empty",
        "drain": "dry",
        "scrape": "disassemble",
        "open up": "open",
        "peel off": "disassemble",
        "deflate": "deflate",
        "install back": "attach",
        "twist off": "unscrew",
        "close up": "close",
        "tear out": "disassemble",
        "bury": "dig",
        "clamp": "fix",
        "crack": "disassemble",
        "drink": "empty",
        "wipe off": "clean",
        "collect": "assemble",
        "shave": "disassemble",
        "wipe": "clean",
        "put": "insert",  # TODO: 'put into' is 'insert', otherwise...
        "put in": "insert",
        "stir": "combine",
        "chop off": "divide",
        "lighten": "turn on",
        "tie": "attach",
        "ski up": "ascend",
        "steam": "heat up",
        "replace": "substitute",
        "load": "insert",
        "unload": "pull out",
        "fit": "assemble",
        "ink": "stain",
        "mark": "stain",
        "color": "stain",
        "remove": "detach",
        "pull up": "ascend",
        "pull down": "descend",
        "put on": "combine",  # TODO: maybe place on top of sm -> makes it higher - should also parse put on differently
        "put off": "remove",
        "install": "assemble",
        "uninstall": "disassemble",
        "slice": "split",
        "wipe ip": "clean",
        "take off": "remove",
        "throw": "push",
        "peel": "disassemble",
        "peel off": "disassemble",
        "pick up": "lift",  # "raise",
        "cut off": "cut",
        "cut away": "remove",
        "chop": "cut",
        "cut": "cut",
        "freeze": "heat down",
        "seal": "close",
        "unseal": "open",
        "sprinkle": "spread",
        "tidy up": "clean",
        "take down": "detach",
        "scrub": "clean",
        "rub": "clean",
        "clean up": "clean",
        "wipe up": "clean",
        "wrap up": "wrap",
        "put down": "drop",
        "paste": "attach",
        "brush": "clean",
        "polish": "clean",
        "roll up": "wrap",
        "simmer": "heat up",
        "crumb": "disassemble",
        "slice up": "cut",
        "warmp up": "heat up",
        "whisk": "combine",
        "bury": "dig",
        "pick": "raise",
        "cut out": "cut",
        "smash": "disassemble",
        "jump": "ascend",
        "patch": "combine",
        "toss": "push",
        "tighten": "attach",
        "crush": "disassemble",
        "peg": "attach",
    }
    return verb2cos.get(verb, verb)
