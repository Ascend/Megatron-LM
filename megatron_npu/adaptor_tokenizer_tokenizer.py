import sys
import megatron
from megatron.tokenizer.tokenizer import _BertWordPieceTokenizer, _GPT2BPETokenizer, _SentencePieceTokenizer, _vocab_size_with_padding


def build_tokenizer(args):
    """Initialize tokenizer."""
    if args.rank == 0:
        print('> building {} tokenizer ...'.format(args.tokenizer_type),
              flush=True)

    if args.tokenizer_type != 'SentencePieceTokenizer':
        assert args.vocab_file is not None

    # Select and instantiate the tokenizer.
    if args.tokenizer_type == 'BertWordPieceLowerCase':
        tokenizer = _BertWordPieceTokenizer(vocab_file=args.vocab_file,
                                            lower_case=True,
                                            vocab_extra_ids=args.vocab_extra_ids)
    elif args.tokenizer_type == 'BertWordPieceCase':
        tokenizer = _BertWordPieceTokenizer(vocab_file=args.vocab_file,
                                            lower_case=False,
                                            vocab_extra_ids=args.vocab_extra_ids)
    elif args.tokenizer_type == 'GPT2BPETokenizer':
        assert args.merge_file is not None
        tokenizer = _GPT2BPETokenizer(args.vocab_file, args.merge_file)
    elif args.tokenizer_type == 'SentencePieceTokenizer':
        assert args.tokenizer_model is not None
        tokenizer = _SentencePieceTokenizer(args.tokenizer_model, vocab_extra_ids=args.vocab_extra_ids)
    else:
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(args.tokenizer_type))

    # Add vocab size.
    if args.vocab_size is not None:
        assert args.vocab_size >= tokenizer.vocab_size
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)
    else:
        args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, args)

    return tokenizer

megatron.tokenizer.tokenizer.build_tokenizer = build_tokenizer

for k, v in sys.modules.items():
    if 'megatron' in k and hasattr(v, 'build_tokenizer'):
        setattr(v, 'build_tokenizer', build_tokenizer)
