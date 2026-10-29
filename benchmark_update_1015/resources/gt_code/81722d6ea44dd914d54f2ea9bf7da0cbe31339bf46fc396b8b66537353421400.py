import networkx as nx
import matplotlib.pyplot as plt

# Define Morse code symbols
dot = "●"
dash = "—"

# Create Morse code mapping for letters
morse_dict = {
    "a": dot + dash,
    "b": dash + dot * 3,
    "c": dash + dot + dash + dot,
    "d": dash + dot * 2,
    "e": dot,
    "f": dot * 2 + dash + dot,
    "g": dash * 2 + dot,
    "h": dot * 4,
    "i": dot * 2,
    "j": dot + dash * 3,
    "k": dash + dot + dash,
    "l": dot + dash + dot * 2,
    "m": dash * 2,
    "n": dash + dot,
    "o": dash * 3,
    "p": dot + dash * 2 + dot,
    "q": dash * 2 + dot + dash,
    "r": dot + dash + dot,
    "s": dot * 3,
    "t": dash,
    "u": dot * 2 + dash,
    "v": dot * 3 + dash,
    "w": dot + dash * 2,
    "x": dash + dot * 2 + dash,
    "y": dash + dot + dash * 2,
    "z": dash * 2 + dot * 2,
}

# Sort by code length and value
sorted_morse = dict(sorted(morse_dict.items(), key=lambda x: (len(x[1]), x[1])))
reverse_map = {v: k for k, v in morse_dict.items()}
reverse_map[""] = ""

# Build prefix tree graph
G = nx.DiGraph()
for char, code in sorted_morse.items():
    predecessor = code[:-1]
    edge_symbol = code[-1]
    G.add_edge(reverse_map[predecessor], char, symbol=edge_symbol)

# Assign layers for layout
for i, level in enumerate(nx.topological_generations(G)):
    for node in level:
        G.nodes[node]["level"] = i

pos = nx.multipartite_layout(G, subset_key="level", align="vertical")
for node in pos:
    pos[node][1] *= -1

# Draw the graph with new styling
plt.figure(figsize=(14, 10))
nx.draw_networkx_nodes(G, pos, node_color='darkblue', node_size=800, alpha=0.9)
nx.draw_networkx_edges(G, pos, edge_color='maroon', width=2, alpha=0.7, arrows=True, arrowsize=20)
nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold', font_size=10)

edge_labels = {(u, v): d['symbol'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkgreen', font_size=9, font_weight='bold')

plt.title("Morse Code Prefix Tree Structure", fontsize=16, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()
plt.show()