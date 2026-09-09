import networkx as nx
import matplotlib.pyplot as plt
from src.engine.board import ADJACENCY, is_supply_center

G = nx.Graph()
for node, neighbors in ADJACENCY.items():
    for n in neighbors:
        G.add_edge(node, n)

POS = nx.kamada_kawai_layout(G)

plt.ion()
FIG, AX = plt.subplots(figsize=(10, 8))

def get_node_colors_and_labels(state):
    node_colors = []
    labels = {}
    for node in G.nodes():
        owner = state.territory_owners.get(node)
        color = 'lightgray'
        if owner:
            color = owner.value.lower()
        node_colors.append(color)
        
        label = node
        if is_supply_center(node):
            label += " *"
        unit = state.get_unit_at(node)
        if unit:
            label += f"\n[{unit.player.name[0]}]"
        labels[node] = label
    return node_colors, labels

def draw_board(state, title_suffix=""):
    AX.clear()
    node_colors, labels = get_node_colors_and_labels(state)

    nx.draw(
        G, pos=POS, ax=AX, 
        node_color=node_colors, 
        labels=labels,
        with_labels=True, 
        node_size=2000,
        font_size=10,
        font_color='black',
        font_weight='bold',
        edgecolors='black',
        linewidths=2
    )
    
    AX.set_title(f"Diplomacy Board - Turn {state.turn} {title_suffix}")
    FIG.canvas.draw()
    FIG.canvas.flush_events()
    plt.pause(0.5)

def draw_action_phase(state, orders, log):
    # First, draw the base board
    AX.clear()
    node_colors, labels = get_node_colors_and_labels(state)

    nx.draw_networkx_nodes(G, pos=POS, ax=AX, node_color=node_colors, node_size=2000, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G, pos=POS, ax=AX, labels=labels, font_size=10, font_weight='bold')
    
    # Draw base edges in light gray
    nx.draw_networkx_edges(G, pos=POS, ax=AX, edge_color='lightgray')
    
    log_text = " ".join(log.events)
    
    # Draw order arrows
    for o in orders:
        if o.order_type.name == "MOVE":
            # Check if succeeded
            success = f"Move succeeds: {o.unit_territory} -> {o.target}" in log.events
            color = 'green' if success else 'red'
            style = 'solid' if success else 'dashed'
            width = 3 if success else 2
            
            # Use matplotlib annotate to draw directed arrows
            if o.target in POS and o.unit_territory in POS:
                AX.annotate("",
                    xy=POS[o.target], xycoords='data',
                    xytext=POS[o.unit_territory], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color=color, ls=style, lw=width, shrinkA=20, shrinkB=20, connectionstyle="arc3,rad=0.1"),
                )
                
        elif o.order_type.name == "SUPPORT":
            cut = any(f"Support cut: {o.unit_territory}" in event for event in log.events)
            color = 'blue'
            style = 'dashed' if cut else 'dotted'
            width = 2
            if o.target in POS and o.unit_territory in POS:
                AX.annotate("",
                    xy=POS[o.target], xycoords='data',
                    xytext=POS[o.unit_territory], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color=color, ls=style, lw=width, shrinkA=20, shrinkB=20, connectionstyle="arc3,rad=-0.1"),
                )

    # Add a legend/log box
    info = "Arrows:\nSolid Green: Success Move\nDashed Red: Bounced Move\nDotted Blue: Support (dashed if cut)"
    AX.text(0.02, 0.98, info, transform=AX.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    AX.set_title(f"Diplomacy Board - Turn {state.turn} (ACTION RESOLUTION)")
    FIG.canvas.draw()
    FIG.canvas.flush_events()
    plt.pause(3.0) # Pause so user can see what happened!

