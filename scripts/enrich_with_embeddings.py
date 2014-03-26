import gzip

__author__ = 'husnusensoy'

lkp = {"-RCB-": "}", "-LCB-": "{", "-RRB-": ")", "-LRB-": "("}

convert = lambda x: lkp[x] if x in lkp else x


skiplist = [')', '(', '}', '{']
import logging

logging.basicConfig(level=logging.INFO)


super_open = lambda f: gzip.open(f) if f.endswith(".gz") else open(f)

def vector_lookup(file, delim='\t', offset=2, vdim=25):
    typev_dict = {}
    with super_open(file) as fp:
        for line in fp:
            tokens = line.split(delim)

            if len(tokens[offset:]) != vdim:
                logging.fatal(
                    "Embedding length(%d) does not confirm with expected length(%d)" % (len(tokens[offset:]), vdim))
                exit(1)

            word = convert(tokens[0])
            type_vect = [float(t) for t in tokens[offset:]]

            typev_dict[word] = type_vect

    return typev_dict


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Create a clone of CoNLL corpus by adding given embeddings',
                                     epilog="python %s best-dis+om.enw.type.gz conll conll_scode "%sys.argv[0])
    parser.add_argument('embeddings_file',
                        help="Delimited embedding file first token to be word others to be the embedding dimensions")
    parser.add_argument('src',
                        help="Root directory of CoNLL corpus to be used as source")
    parser.add_argument('target',
                        help="Root directory of CoNLL corpus to be used as target")
    parser.add_argument('--delimiter', default='\t', help='File delimiter. Default is <TAB>')

    parser.add_argument('--offset', default=1, type=int, help='Starting offset of embedding token. Default: 1')

    #parser.add_argument('--skipbraces',  default=False, action='store_true',
    #                    help='Do not generate bracket embeddings. Instead use default embedding')

    parser.add_argument('--unk_as_default', default=False, action='store_true',
                        help='Use <unk> embedding as the default embedding. Otherwise 0 embedding is used')

    parser.add_argument('--token', default=False, action='store_true',
                        help='Token based embeddings instead of type based embeddings')

    parser.add_argument('--unk_key',default="*UNKNOWN*", help="UNK key to be used to replace unknown words")
    parser.add_argument('--length', type=int, default=25, help='Expected length of embeddings')

    args = parser.parse_args()

    logging.info(str(args))

    if args.token:
        viter = super_open(args.embeddings_file)
    else:
        vmap = vector_lookup(file=args.embeddings_file, delim=args.delimiter, vdim=args.length, offset=args.offset)

    import os

    for dir in sorted(os.listdir(args.src)):
        logging.info(dir)
        for file in sorted(os.listdir("%s/%s" % (args.src, dir))):
            logging.info("\t%s" % file)

            if not os.path.exists("%s/%s" % (args.target, dir)):
                os.makedirs("%s/%s" % (args.target, dir))

            with open("%s/%s/%s" % (args.src, dir, file)) as fp, open("%s/%s/%s" % (args.target, dir, file), "w") as wp:
                for line in fp:
                    if len(line.strip()) == 0:
                        print >> wp, ""
                    else:
                        tokens = line.strip().split('\t')

                        if args.token:
                            embedding_token = viter.next().strip().split(args.delimiter)

                            if embedding_token[0] != tokens[1]:
                                logging.fatal("Embedding (%s) file is not aligned to corpus content(%s)",
                                              embedding_token[0] != tokens[1])
                                exit(1)
                            else:
                                embedding_v = " ".join(d for d in embedding_token[args.offset:])
                        else:

                            if tokens[1] in vmap:
                                embedding_v = " ".join(str(d) for d in vmap[tokens[1]])
                            else:
                                if args.unk_as_default:
                                    embedding_v = " ".join(str(d) for d in vmap[args.unk_key])
                                else:
                                    embedding_v = " ".join(str(d) for d in [0.]*args.length)

                        print >> wp, "%s\t%s" % (line.strip(), embedding_v)
