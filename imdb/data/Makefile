%.tsv:
	wget "https://datasets.imdbws.com/$@.gz"
	gunzip "$@.gz"

all: title.principals.tsv name.basics.tsv title.ratings.tsv title.en.only_genre.tsv

title.en.only_genre.tsv: title.basics.tsv
	cd .. && python -m which_genre.data_prep --filter_en data/$<

wordlist:
	wget -O wordlist https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa.txt
