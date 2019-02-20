import pygraphviz as pgv
import numpy as np
import sys

def construct_block(graph, myid, ops):

    ops_name = ["tanh", "ReLU", "identity", "sigmoid"]

    graph.add_node(myid,
                   label="{}".format(ops_name[ops[1]]),
                   color='black',
                   fillcolor='yellow',
                   shape='box',
                   style='filled')

def connect_block(graph, myid, ops, output_used):

    if ops[0] == -1:
        graph.add_edge("ht-1", myid)
        graph.add_edge("xt", myid)
    else:
        graph.add_edge(ops[0], myid)
    output_used.append(ops[0])

def creat_graph(cell_arc):

    G = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open', rankdir='TD')

    #creat input
    G.add_node("ht-1", label="H[t-1]", color='black', shape='box')
    G.add_node("xt", label="x[t]", color='black', shape='box')
    G.add_subgraph(["ht-1", "xt"], name='cluster_inputs', rank='same', rankdir='TD', color='white')

    #creat blocks
    for i in range(0, len(cell_arc)):
        construct_block(G, i, cell_arc[i])

    #connect blocks to each other
    output_used = []
    for i in range(0, len(cell_arc)):
        connect_block(G, i, cell_arc[i], output_used)

    #creat output
    G.add_node("avg",
               label="avg",
               color='black',
               fillcolor='green',
               shape='box',
               style='filled')
    G.add_node("ht",
               label="h[t]",
               color='black',
               fillcolor='pink',
               shape='box',
               style='filled')
    G.add_edge("avg", "ht")
    G.add_edge("ht", "ht-1")
    
    for i in range(0, len(cell_arc)):
        if not i in output_used:
            G.add_edge(i, "avg")

    return G


def main():

    if(len(sys.argv) <= 1):
        rnn_cell = "-1 0 0 0 1 0 1 1 2 0 2 0 4 0 6 0 6 0 7 1 7 0 9 0"
    else:
        rnn_cell = "-1 "
        for i in range(1, len(sys.argv)/2+1):
            rnn_cell += "{} ".format(sys.argv[i])
        print("{}".format(rnn_cell))

    rcell = np.array([int(x) for x in rnn_cell.split(" ") if x])

    rcell = np.reshape(rcell, [-1, 2])

    Gr = creat_graph(rcell)

    Gr.write("rnncell.dot")

    vizGr = pgv.AGraph("rnncell.dot")

    vizGr.layout(prog='dot')

    vizGr.draw("rnncell.png")


if __name__ == '__main__':
    main()
