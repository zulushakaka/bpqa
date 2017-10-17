from sparql_backend import backend
from string import Template
import logging
from knowledge_graph import KnowledgeGraph
from entity_linker import WebQOracleLinker

logger = logging.getLogger(__name__)

sparql_backend = backend.SPARQLHTTPBackend('202.120.38.146', '8699', '/sparql')

query_tmpl1 = Template('''
PREFIX fb: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?p ?t
WHERE {
    fb:${e} ?p ?x .
    ?p fb:type.property.expected_type ?t .
    FILTER (fb:${e} != ?x)
}
''')

query_tmpl2 = Template('''
PREFIX fb: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?p1 ?p2 ?t1 ?t2 
WHERE {
    fb:${e} ?p1 ?x1 .
    ?x1 ?p2 ?x2 .
    ?p1 fb:type.property.expected_type ?t1 .
    ?p2 fb:type.property.expected_type ?t2 .
    FILTER (fb:${e} != ?x2)
}
''')


def crawl_one_hop(seed):
    query = query_tmpl1.substitute(e=seed)
    results = sparql_backend.query(query)
    if not results:
        logger.debug('Error in SPARQL query:\n%s', query)
    # print(results)
    return results


def crawl_two_hop(seed):
    query = query_tmpl2.substitute(e=seed)
    results = sparql_backend.query(query)
    if not results:
        logger.debug('Error in SPARQL query:\n%s', query)
    # print(results)
    return results


def retrieve(seeds):
    crawled_one_hop = [crawl_one_hop(seed) for seed in seeds]
    crawled_two_hop = [crawl_two_hop(seed) for seed in seeds]
    kg = KnowledgeGraph()
    for crawled_from_seed, seed in zip(crawled_one_hop, seeds):
        kg.add_node(seed)
        for path in crawled_from_seed:
            x = kg.add_node()
            kg.add_edge(seed, x, path[0])
    for crawled_from_seed, seed in zip(crawled_two_hop, seeds):
        for path in crawled_from_seed:
            predge = kg.find_edge(seed, None, path[0])
            if predge:
                med = predge.split('--')[2]
                x = kg.add_node()
                kg.add_edge(med, x, path[1])
            else:
                x1 = kg.add_node()
                x2 = kg.add_node()
                kg.add_edge(seed, x1, path[0])
                kg.add_edge(x1, x2, path[1])

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
    kg = retrieve([topic_ent])
    kg.show()
