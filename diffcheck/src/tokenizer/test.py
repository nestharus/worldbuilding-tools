import spacy

legal_pos_dep_pairs = {
    # Nouns and Proper Nouns
    ("NOUN", "nsubj"),  # Noun as nominal subject
    ("NOUN", "nsubjpass"),  # Noun as passive nominal subject
    ("NOUN", "dobj"),  # Noun as direct object
    ("NOUN", "pobj"),  # Noun as object of preposition
    ("NOUN", "attr"),  # Noun as predicate attribute
    ("NOUN", "appos"),  # Noun as appositive modifier
    ("NOUN", "nmod"),  # Noun as nominal modifier
    ("NOUN", "poss"),  # Noun as possessive modifier
    ("NOUN", "compound"),  # Noun in compound noun
    ("NOUN", "conj"),  # Noun as part of a conjunction
    ("NOUN", "dep"),  # Noun as unspecified dependent
    ("NOUN", "csubj"),  # Noun in clausal subject role
    ("NOUN", "csubjpass"),  # Noun as passive clausal subject
    ("NOUN", "oprd"),  # Noun as object predicate

    ("PROPN", "nsubj"),  # Proper noun as nominal subject
    ("PROPN", "nsubjpass"),  # Proper noun as passive nominal subject
    ("PROPN", "dobj"),  # Proper noun as direct object
    ("PROPN", "pobj"),  # Proper noun as object of preposition
    ("PROPN", "attr"),  # Proper noun as predicate attribute
    ("PROPN", "appos"),  # Proper noun as appositive
    ("PROPN", "nmod"),  # Proper noun as nominal modifier
    ("PROPN", "poss"),  # Proper noun as possessive modifier
    ("PROPN", "compound"),  # Proper noun in compound
    ("PROPN", "conj"),  # Proper noun in conjunction

    # Adjectives
    ("ADJ", "amod"),  # Adjective as adjectival modifier
    ("ADJ", "acomp"),  # Adjective as adjectival complement
    ("ADJ", "attr"),  # Adjective as predicate attribute
    ("ADJ", "oprd"),  # Adjective as object predicate
    ("ADJ", "conj"),  # Adjective in conjunction

    # Determiners
    ("DET", "det"),  # Determiner as noun determiner
    ("DET", "predet"),  # Predeterminer (e.g., "all the books")

    # Pronouns
    ("PRON", "nsubj"),  # Pronoun as subject
    ("PRON", "nsubjpass"),  # Pronoun as passive nominal subject
    ("PRON", "dobj"),  # Pronoun as direct object
    ("PRON", "pobj"),  # Pronoun as object of preposition
    ("PRON", "poss"),  # Pronoun as possessive modifier
    ("PRON", "attr"),  # Pronoun as predicate attribute
    ("PRON", "appos"),  # Pronoun as appositive
    ("PRON", "oprd"),  # Pronoun as object predicate

    # Verbs
    ("VERB", "ROOT"),  # Main verb as root of sentence
    ("VERB", "aux"),  # Verb as auxiliary
    ("VERB", "auxpass"),  # Auxiliary in passive voice
    ("VERB", "xcomp"),  # Verb as open clausal complement
    ("VERB", "ccomp"),  # Verb as clausal complement with subject
    ("VERB", "advcl"),  # Verb in adverbial clause
    ("VERB", "acl"),  # Verb in relative clause
    ("VERB", "relcl"),  # Verb in relative clause modifier
    ("VERB", "conj"),  # Verb as part of conjunction
    ("VERB", "oprd"),  # Verb as object predicate

    # Adverbs
    ("ADV", "advmod"),  # Adverb as adverbial modifier
    ("ADV", "npmod"),  # Adverb modifying a noun phrase
    ("ADV", "oprd"),  # Adverb as object predicate
    ("ADV", "conj"),  # Adverb in conjunction

    # Numbers
    ("NUM", "nummod"),  # Number as numeral modifier
    ("NUM", "quantmod"),  # Number as quantifier modifier

    # Conjunctions
    ("CCONJ", "cc"),  # Coordinating conjunction
    ("SCONJ", "mark"),  # Subordinating conjunction as clause marker

    # Prepositions
    ("ADP", "prep"),  # Preposition
    ("ADP", "pcomp"),  # Prepositional complement
    ("ADP", "case"),  # Case marking element (e.g., "of" in "the best of")

    # Particles
    ("PART", "prt"),  # Particle in phrasal verb
    ("PART", "advmod"),  # Particle as adverbial modifier

    # Auxiliary Verbs
    ("AUX", "aux"),  # Auxiliary verb
    ("AUX", "auxpass"),  # Auxiliary for passive construction

    # Clauses
    ("X", "mark"),  # Other (e.g., "to" in infinitival clause)
    ("X", "advmod"),  # Other as adverbial modifier
}

nlp = spacy.load("en_core_web_sm")

compounds = [
    "merry-go-round",
    "x-ray",
    "ice-cream",
    "cost-benefit",
    "parent-child",
    "client-server",
    "jan-dec",
    "7-Eleven",
    "Dr. J. Smith",
    "A. B. Smith",
    "January - March",
    "april-december",
    "1-7",
    "ten-four",
    "ninety-nine",
]

# for compound in compounds:
#     print("Checking ", compound)
#     doc = nlp(compound)
#     illegals = []
#     for token in doc:
#         if (token.pos_, token.dep_) not in legal_pos_dep_pairs:
#             illegals.append(f"\tILLEGAL {token.text}: {token.pos_} {token.dep_} -> {token.head.text}")
#     print('LEGAL' if len(illegals) == 0 else 'ILLEGAL')
#     for illegal in illegals:
#         print(illegal)

print('Analyzing "Going up, I saw it"')

doc = nlp('Going up, I saw it.')
for token in doc:
    if (token.pos_, token.dep_) not in legal_pos_dep_pairs:
        print(f'ILLEGAL TOKEN {token.text}: {token.pos_} {token.dep_} -> {token.head.text}')
    else:
        print(f"{token.text}: {token.pos_} {token.dep_} -> {token.head.text}")