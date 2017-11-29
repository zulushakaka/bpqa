from HR_BiLSTM.model import HRBiLSTM
import json
from string import Template
import numpy as np
from sparql import *


def crawl_one_hop(seed):
    query = Template('''
    PREFIX fb: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?p
    WHERE {
        fb:${e} ?p ?x .
        FILTER (fb:${e} != ?x)
    }
    ''').substitute(e=seed)
    result = sparql_backend.query(query)
    return result


if __name__ == '__main__':
    model = HRBiLSTM()
    # model.train(2, 'HR_BiLSTM/model-6.ckpt')
    model.load_model('HR_BiLSTM/model_b32h200-6.ckpt')

    # eval on test
    with open('data/WebQSP/WebQSP.test.json') as f:
        test = json.load(f)
        correct = 0
        count = 0

        q_count = 0
        for q in test['Questions']:
            print q_count,
            q_count += 1
            topic_entity = q['Parses'][0]['TopicEntityMid']
            inf_chain = q['Parses'][0]['InferentialChain']

            if not inf_chain or not topic_entity:
                continue

            candidates = set()
            cands = crawl_one_hop(topic_entity)
            for path in cands:
                candidates.add(path[0])
                # candidates.add(path[1])
            candidates = list(candidates)

            if len(candidates) > 32:
                candidates = candidates[:32]
            for gold in inf_chain:
                if not gold in candidates:
                    candidates.append(gold)

            best_score = 0
            best_cand = None
            for cand in candidates:
                if not cand in model.rel2idx:
                    continue
                score = model.predict(q['RawQuestion'][:-1], cand)
                if score > best_score:
                    best_score = score
                    best_cand = cand
            if best_cand in inf_chain:
                correct += 1
                print('t')
            else:
                print('f')
            count += 1

        print(correct,'/',count, float(correct)/count)