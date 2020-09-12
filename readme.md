# Research on a vowel classifier.

This project aims to create a phone (as in phoneme) classifier for use in human computer
interaction.

The best voice to text transcription I've encountered is somewhat awkward and
unreliable. General phone classificaion a difficult problem.

This project aims to implement a classifier for a subset of human pronouncable
phones. To start, only temporally agnostinc phones (like vowels) will be
supported. This approach restricts problem scope and hopefully will increase
reliablility and speed during human computer interaction.

An alphabet will be selected such that no two phones in the alphabet are confusable,
either by the classifier, or by speakers of popular languages.

For now the alphabet is five phones: a, æ, i, and o. ("e" is used to represent "æ" in the test data).

## Related links

- https://phoible.org
- http://www.ipachart.com
- https://en.wikipedia.org/wiki/International_Phonetic_Alphabet

# Try it for yourself!

You'll need [pipenv](https://pipenv-fork.readthedocs.io/en/latest/) and [just](https://github.com/casey/just).

Run `just go` to start training and testing.

To contribute your own training data, run `just record`, `just check`, then either `just approve` or
`just reject`. PRs welcome.
