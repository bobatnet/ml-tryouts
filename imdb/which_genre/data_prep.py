import click
import re

from .params import fname, fname_wordlist

@click.command()
@click.option('--filter_en', type=click.Path(exists=True))
@click.option('--save', default=fname, type=click.Path(writable=True))
def filter_titles(filter_en, save):
    en_words = map(lambda x: x.strip(), open(fname_wordlist).readlines())
    any_en = re.compile('|'.join(map(r'\b{}\b'.format, en_words)))

    j = 0
    inp_col, target_col = None, None

    with open(save, 'w') as outfile:
        with open(filter_en) as inpfile:
            for i, ln in enumerate(inpfile):
                ln_ = ln.strip().split(sep='\t')

                if i:
                    inp = ln_[inp_col]
                    if any_en.search(inp):
                        target = ln_[-1]
                        outfile.write(f'{inp}\t{target}\n'.lower())
                        j += 1
                else:
                    inp_col, target_col = ln_.index('primaryTitle'), ln_.index('genres')

    print(f'Saved {j} records out of {i}')

if __name__ == '__main__':
    filter_titles()
