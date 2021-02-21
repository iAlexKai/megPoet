import torch
from fairseq.models.bart.bart_model import BARTModel

bpe_json = '../data_process/poet/resources/encoder.json'

epoch = "_best"

bart = BARTModel.from_pretrained(
    '../checkpoints/',
    checkpoint_file='checkpoint{}.pt'.format(epoch),
    data_name_or_path='../data-bin/poet',
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 32


def _load_dict(json_path):
    import json
    with open(json_path, 'r') as file:
        ind2char = {}
        word_dict = json.loads(file.read())
        for (key, val) in word_dict.items():
            ind2char[val] = key
        return ind2char


ind2char = _load_dict(bpe_json)
import time
cur_time = time.strftime("%m_%d_%H_%M_%S")

if epoch.startswith('_'):
    epoch = epoch[1:]

# with open('../data_process/poet/test.source') as src, open('../output/poet/test_{}_{}.hypo'.format(cur_time, epoch), 'w') as fout:
with open('../data_process/poet/test.source') as src, open('../output/poet/test_{}_{}.hypo'.format(cur_time, epoch), 'w') as fout:
    for sline in src:

        cur_input = sline.strip().split(' ')
        # import pdb
        # pdb.set_trace()
        title_index = cur_input.index('5')
        title_tokens = cur_input
        title_str = " ".join(title_tokens)

        # import pdb
        # pdb.set_trace()
        cur_poem = ["".join(ind2char[int(i)] for i in title_tokens)]
        next_input = sline.strip()  # str
        with torch.no_grad():
            for _ in range(4):
                # import pdb
                # pdb.set_trace()
                hypotheses_batch = bart.sample(next_input, beam=4, lenpen=2.0, max_len_b=30, min_len=5)

                next_input = title_str + ' 4 ' + " ".join([str(i) for i in hypotheses_batch[:-1]])
                cur_poem.append("".join(ind2char[i] for i in hypotheses_batch[:-1]))
            print("\n".join(cur_poem))
            fout.write("\n".join(cur_poem) + '\n')
            fout.flush()