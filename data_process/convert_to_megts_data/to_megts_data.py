data = open('train_data.txt', 'r').read().strip().split('\n')
out = open('all.txt', 'w')

train_corpus = []
for line in data:
    content = line.split('\t')
    title = content[0]
    condidates = [title] + [content[-1][:8]] + [content[-1][8:16]] + [content[-1][16:24]] + [content[-1][24:]]
    # import pdb
    # pdb.set_trace()

        
    for i in range(4):
        train_corpus.append("{}@{}\t{}".format(title, condidates[i], condidates[i+1]))

for i in range(len(train_corpus)):
    out.write('{}\n'.format(train_corpus[i]))