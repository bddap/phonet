#!/usr/bin/env python3

import json
import sys

table = json.load(open("sounds.json"))
order = json.load(open("src_order.json"))

def lookup(ipa_letter):
    matches = [t for t in table if t['ipa_symbol'] == ipa_letter]
    assert len(matches) == 1
    return matches[0]

json.dump([lookup(letter)['name'] for letter in order], sys.stdout)


