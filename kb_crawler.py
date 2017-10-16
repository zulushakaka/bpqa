from sparql_backend import backend
from string import Template
import logging
from knowledge_graph import KnowledgeGraph
from entity_linker import WebQOracleLinker

logger = logging.getLogger(__name__)

backend = None


def init_sparql_backend(back):
    if not back:
        back = backend.SPARQLHTTPBackend('202.120.38.146', '8699', '/sparql')


init_sparql_backend(backend)

query_tmpl = Template(
'''
PREFIX fb: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?pred1 ?x1 ?pred2 ?x2
WHERE {
    fb:${e} ?pred1 ?x1 .
    ?x1 ?pred2 ?x2 .
    FILTER (fb:${e} != ?x2)
}
''')


def crawl_two_hop(seed):
    query = query_tmpl.substitute(e=seed)
    results = backend.query(query)
    if not results:
        logger.debug('Error in SPARQL query:\n%s', query)
    return results


def retrieve(seeds):
    crawled_two_hop = [crawl_two_hop(seed) for seed in seeds]
    kg = KnowledgeGraph()
    for crawled_from_seed, seed in zip(crawled_two_hop, seeds):
        rels = set([(path[0], path[2]) for path in crawled_from_seed])
        kg.add_node(seed)
        kg.nodes[seed].add_candidates([seed])
        for rel in rels:
            candidates = filter(lambda x: (x[0], x[2]) == rel , crawled_from_seed)
            x1_cand = [x[1] for x in candidates]
            x2_cand = [x[3] for x in candidates]

            x1 = kg.add_node()
            x2 = kg.add_node()
            kg.add_edge(seed, x1, rel[0])
            kg.add_edge(x1, x2, rel[1])

            x1.add_candidates(x1_cand)
            x2.add_candidates(x2_cand)

    return kg


def merge(kg):
    # add 'equal' between nodes with overlapping candidates, merge nodes with same candidates
    for key, node in kg.items():
        for key2, node2 in kg.items():
            if key == key2:
                continue
            elif node.candidates == node2.candidates:
                kg.merge_node(key, key2)
            elif node.candidates & node2.candidates:
                kg.add_edge(key, key2, '*equal*')


def enhance(kg, literals):
    # add numerical relations
    for literal in literals:
        kg.add_node(literal)
        for key, node in kg.items():
            if 'data_time' in node.type:
                kg.add_edge(key, literal, '*numerical*')


if __name__ == '__main__':
    q = 'what is the name of justin bieber brother?'
    linker = WebQOracleLinker()
    topic_ent = linker.link(q)
    print(topic_ent)
