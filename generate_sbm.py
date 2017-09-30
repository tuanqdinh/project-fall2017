import graph_tool.all as gt
import graph_tool

g = gt.collection.data["polblogs"]
g = gt.GraphView(g, vfilt=gt.label_largest_component(g))
g = gt.Graph(g, prune=True)
state = gt.minimize_blockmodel_dl(g)
u = gt.generate_sbm(state.b.a, gt.adjacency(state.get_bg(),state.get_ers()),
g.degree_property_map("out").a,
g.degree_property_map("in").a, directed=True)

a = graph_tool.spectral.adjacency(g)
A = a.todense()
from IPython import embed; embed()
