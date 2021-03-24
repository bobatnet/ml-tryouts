%.tsv:
	wget "https://datasets.imdbws.com/$@.gz"
	gunzip "$@.gz"

all: title.principals.tsv name.basics.tsv title.ratings.tsv title.basics.tsv

genre_from_title: title.basics.tsv

wordlist:
	wget -O wordlist https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa.txt

.PHONY: genre_from_title
