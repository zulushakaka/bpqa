import logging

logger = logging.getLogger(__name__)


class KnowledgeGraph (object):
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.var_count = 0
        
    def add_node(self, name=None):
        if name:
            self.nodes[name] = Node(grounde=True, name=name)
        else:
            var_name = 'var%d' % self.var_count
            self.nodes[var_name] = Node(grounded=False, name=var_name)
        return name

    def add_edge(self, left, right, name):
        if not self.nodes.has_key(left):
            logging.debug('Error: node %s not found', left)
        if not self.nodes.has_key(right):
            logging.debug('Error: node %s not found', right)
        self.edges['%s_%s_%s' % (left.name, name, right.name)] = Edge(grounded=True, name=name, left=left, right=right)

    def merge_node(self, n1, n2):
        for edge in self.edges.values():
            if set((edge.left.name, edge.right.name)) == set((n1, n2)):
                del self.edges[str(edge)]
            elif edge.left == n2:
                edge.left.name


class Node (object):
    def __init__(self, grounded, name):
        self.grounded = grounded
        self.name = name
        self.candidates = set()

    def add_candidates(self, cand):
        self.candidates.update(cand)

    def __repr__(self):
        return self.name


class Edge (object):
    def __init__(self, grounded, name, left, right):
        self.grounded = grounded
        self.name = name
        self.left = left
        self.right = right
        self.candidates = set()

    def __repr__(self):
        return '%s_%s_%s' % (self.left.name, self.name, self.right.name)
