import json


class EntityLinker (object):
    def link(self, q):
        pass


class WebQOracleLinker (EntityLinker):
    def __init__(self, part='train'):
        self.oracle = {}
        with open('data/WebQSP/WebQSP.%s.json' % part, 'r') as f:
            data = json.load(f)
            questions = data['Questions']
            for question in questions:
                q = question['RawQuestion']
                parse = question['Parses'][0]
                topic_entity = parse['TopicEntityMid']
                self.oracle[q] = topic_entity

    def link(self, q):
        return self.oracle[q]