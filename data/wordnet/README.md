# Chinese Open Wordnet (COW)

Upstream: https://github.com/omwn/omw-data/tree/19256777c557e400dd5ef592428fa78def6c07dc/wns/cow

`wn-data-cmn.tab` and `LICENSE` are unmodified upstream files. `source.json`
records the fixed upstream commit, download URLs, byte sizes and SHA-256 hashes.
The original WordNet license and attribution are retained in `LICENSE`.

The application reads this tab-separated resource directly with Python's standard
library. NLTK and runtime corpus downloads are unnecessary. It keeps separate
synset identifiers and parts of speech, selecting Chinese expressions of 2–32
characters and synsets with at least two eligible expressions. Expressions in a
shared synset are retrieval alternatives; contextual equivalence is not guaranteed.

`../query_expansion_overrides.json` contains separately attributed, reviewed
project corrections. These are not presented as upstream WordNet assertions.
Book-specific corrections and exclusions are stored in each saved index.

All normalized groups, project corrections, configuration, and actual book
expression spans are embedded in exported traces for independent offline replay.
