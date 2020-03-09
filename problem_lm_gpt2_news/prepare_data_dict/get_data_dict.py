# -*- coding:utf-8 -*-


import json,os

f='xaa'
from problem_util_yr.loadDict.read_json_tool import read_json
gene=read_json(f)

writer=open('text.txt','w')
vocabset=set()
import re
p_space=re.compile('\s+')


llen=[]


def remove_space(line):
    return re.sub(p_space,'',line)


for f in os.listdir('../rawdata/')[:]:
    if 'json' not in f:continue
    f=os.path.join('../rawdata/',f)
    print (f)
    gene=read_json(f)
    for d in gene:
        title=d['title'][0].strip()
        title=remove_space(title)

        writer.write(title)
        for w in title:
            vocabset.add(w)
        ####
        text=d['content']
        llen.append(len(text))
        writer.write(text+'\n')
        for w in text:
            vocabset.add(w)


        #if len(vocabset)>1000:break
####

ll=list(vocabset)
dic=dict(zip(ll,list(range(5,5+len(ll)))))
dic['PAD']=0
dic['EOS']=1
dic['UNK']=2
dic['unuse3']=3
dic['unuse4']=4

writer=open('vocab.json','w')
writer.write(json.dumps(dic,ensure_ascii=False))

import numpy as np
r1,r2=np.histogram(llen)
print (r1)
print ([int(v) for v in r2])

# [26180  1455   131    29    12     5     3     1     0     1]
# [16, 2948, 5880, 8812, 11744, 14676, 17608, 20540, 23472, 26404, 29337]