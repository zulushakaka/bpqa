from HR_BiLSTM.model import HRBiLSTM
import json
from kb_crawler import crawl_two_hop
import numpy as np


if __name__ == '__main__':
    model = HRBiLSTM()
    model.train(2, 'HR_BiLSTM/model-6.ckpt')

    # eval on test
    with open('data/WebQSP/WebQSP.test.json') as f:
        test = json.load(f)

        correct = 0
        count = 0

        for q in test:
            topic_entity = q['Parses']['TopicEntityMid']
            inf_chain = q['Parses']['InferentialChain']

            candidates = set()
            cands = crawl_two_hop(topic_entity)
            for path in cands:
                candidates.add(path[0])
                candidates.add(path[1])
            candidates = list(candidates)

            best_score = 0
            best_cand = None
            for cand in candidates:
                score = model.predict(q['RawQuestion'][:-1], cand)
                if score > best_score:
                    best_score = score
                    best_cand = cand
            if best_cand in inf_chain:
                correct += 1
            count += 1

        print(correct,'/',count, float(correct)/count)