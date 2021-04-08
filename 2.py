def miniAndMaxiDegrees(graph):
    # each line of graph represents a directed edge

    in_degree = {}
    out_degree = {}
    for g in graph:
        out_degree[g[0]] = out_degree.get(g[0], 0) + 1
        in_degree[g[1]] = in_degree.get(g[1], 0) + 1

    in_nodes = list(in_degree.keys())
    out_nodes = list(out_degree.keys())
    in_nodes_null = set(out_nodes) - set(in_nodes)  # nodes that do not have input
    out_nodes_null = set(in_nodes) - set(out_nodes) # nodes that do not have output
    # set these nodes as zeros
    for i in in_nodes_null:
        in_degree[i] = in_degree.get(i, 0)
    for i in out_nodes_null:
        out_degree[i] = out_degree.get(i, 0)

    in_degree_values = list(in_degree.values())
    min_in_degree = min(in_degree_values)
    max_in_degree = max(in_degree_values)
    out_degree_values = list(out_degree.values())
    min_out_degree = min(out_degree_values)
    max_out_degree = max(out_degree_values)
    return min_in_degree, max_in_degree, min_out_degree, max_out_degree
