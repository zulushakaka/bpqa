from entity_linker import *
from sparql import *
from kb_crawler import *

class Parser:
    def __init__(self):
        self.linker = WebQOracleLinker()
        self.sparql = sparql_backend

    def parse(self, q):
        entities = self.linker.link(q)
        kg = retrieve(entities)
        merge(kg)
        # eqs = kg.find_edge(None, None, '*equal*')
        # eqvars = set()
        # for eq in eqs:
            # print(eq)
            # left, _, right = eq.split('--')
            # eqvars.add(left)
            # eqvars.add(right)
        # print(kg)
        # print(len(eqvars))
        # kg.show()



if __name__ == '__main__':
    parser = Parser()
    parser.parse('what character did natalie portman play in star wars?')