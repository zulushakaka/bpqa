import logging

logger = logging.getLogger(__name__)


class KnowledgeGraph (object):
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.var_count = 0
        
    def add_node(self, name=None):
        if name:
            self.nodes[name] = Node(grounded=True, name=name)
            self.nodes[name].add_candidates([name])
        else:
            name = 'var%d' % self.var_count
            self.nodes[name] = Node(grounded=False, name=name)
            self.var_count += 1
        return name

    def add_edge(self, left, right, name):
        if not left in self.nodes:
            logging.debug('Error: node %s not found', left)
        if not right in self.nodes:
            logging.debug('Error: node %s not found', right)
        left = self.nodes[left]
        right = self.nodes[right]
        self.edges['%s--%s--%s' % (left.name, name, right.name)] = Edge(grounded=True, name=name, left=left, right=right)

    def find_edge(self, left, right, name):
        for key in self.edges:
            key_left, key_name, key_right = key.split('--')
            if key_name == name and (not left or key_left == left) and (not right or key_right == right):
                return key
        return None

    def merge_node(self, n1, n2):
        if self.nodes[n2].grounded:
            if self.nodes[n1].grounded:
                logger.debug('Error: merging two grounded nodes.')
                return
            n1, n2 = n2, n1
        # replace n2 by n1 in all edges
        for key, edge in self.edges.items():
            if set((edge.left.name, edge.right.name)) == set((n1, n2)):
                del self.edges[key]
            elif edge.left.name == n2:
                self.add_edge(n1, edge.right.name, edge.name)
                del self.edges[key]
            elif edge.right.name == n2:
                self.add_edge(edge.left.name, n1, edge.name)
                del self.edges[key]
        del self.nodes[n2]

    def __repr__(self):
        return '<Knowledge graph with %d nodes and %d edges>' % (len(self.nodes), len(self.edges))

    def show(self):
        print('####')
        buffer = []
        for key, edge in self.edges.items():
            if '*equal*' in key:
                buffer.append(key)
            else:
                print(key)
        for key in buffer:
            print(key)
        print(self)


class Node (object):
    def __init__(self, grounded, name):
        self.grounded = grounded
        self.name = name
        self.candidates = set()
        self.type = None

    def add_candidates(self, cand):
        self.candidates.update(cand)

    def set_type(self, t):
        self.type = t

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
        return '%s--%s--%s' % (self.left.name, self.name, self.right.name)
