import json
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def load_layout(scene_dir):
    layout_files = sorted(Path(scene_dir).glob('*layout_data*.json'))
    if not layout_files:
        raise FileNotFoundError(f"No layout_data*.json in {scene_dir}")
    with open(layout_files[0]) as f:
        data = json.load(f)
    return data


def load_arrangement(scene_dir, fixture_name):
    p = Path(scene_dir) / f'{fixture_name}.json'
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def get_products_from_arrangement(arr):
    if arr is None:
        return []
    products = []
    for node in arr['graph']:
        obj_name = node[1]
        if '/' not in obj_name and 'SHELF' not in obj_name:
            product = obj_name.split(':')[0]
            products.append(product)
    return products


def draw_shelf_rect(ax, x, y, l, w, orientation, color, edgecolor,
                    label_line1, label_line2='', fontsize=6, label_color='white', alpha=0.85):
    if orientation == 'vertical':
        draw_l, draw_w = w, l
    else:
        draw_l, draw_w = l, w

    rect_x = x - draw_l / 2
    rect_y = y - draw_w / 2

    rect = FancyBboxPatch(
        (rect_x, rect_y), draw_l, draw_w,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor=edgecolor, linewidth=1.8, alpha=alpha
    )
    ax.add_patch(rect)

    if label_line2:
        ax.text(x, y + 0.06, label_line1, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=label_color)
        ax.text(x, y - 0.06, label_line2, ha='center', va='center',
                fontsize=fontsize - 1, color=label_color, alpha=0.9)
    else:
        ax.text(x, y, label_line1, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=label_color)

    return rect_x, rect_y, draw_l, draw_w


def draw_robot_arrow(ax, x, y, angle_deg=0, length=0.6, color='#ff6600'):
    angle_rad = np.radians(angle_deg)
    dx = length * np.cos(angle_rad)
    dy = length * np.sin(angle_rad)

    ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=3,
                                mutation_scale=20))

    circle = plt.Circle((x, y), 0.2, facecolor=color, edgecolor='black',
                         linewidth=2, zorder=10)
    ax.add_patch(circle)
    ax.text(x, y, 'R', ha='center', va='center', fontsize=7,
            fontweight='bold', color='white', zorder=11)

    ax.text(x, y - 0.45, f'Robot ({x:.1f}, {y:.1f})',
            ha='center', va='top', fontsize=7, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      alpha=0.85, edgecolor=color),
            zorder=11)

    heading_label = f'Heading: {angle_deg} deg'
    if 0 <= angle_deg < 90:
        heading_label += ' (→)'
    elif 90 <= angle_deg < 180:
        heading_label += ' (↑)'
    elif 180 <= angle_deg < 270:
        heading_label += ' (←)'
    else:
        heading_label += ' (↓)'
    ax.text(x, y - 0.75, heading_label,
            ha='center', va='top', fontsize=6, color='#aa4400',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='#fff0e0',
                      alpha=0.8, edgecolor='#aa4400'),
            zorder=11)


def generate_map(scene_dir, output_path=None, robot_angle=0):
    data = load_layout(scene_dir)
    ld = data['layout_data']
    size_x = data['size_x']
    size_y = data['size_y']

    fig, ax = plt.subplots(1, 1, figsize=(28, 18))
    ax.set_xlim(-1.0, size_x + 1.0)
    ax.set_ylim(-1.0, size_y + 1.0)
    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=14, fontweight='bold')
    ax.set_title('Restock Scene — Top-Down Map for Path Planning\n'
                 '(1:1 scale, all coordinates in meters)',
                 fontsize=18, fontweight='bold', pad=20)

    ax.set_facecolor('#f5f0e8')
    fig.patch.set_facecolor('#e8e0d0')

    for x_val in np.arange(0, size_x + 0.1, 1.0):
        ax.axvline(x=x_val, color='#d0c8b8', linewidth=0.5, linestyle='-', alpha=0.6)
    for y_val in np.arange(0, size_y + 0.1, 1.0):
        ax.axhline(y=y_val, color='#d0c8b8', linewidth=0.5, linestyle='-', alpha=0.6)

    for x_val in np.arange(0, size_x + 0.1, 0.5):
        ax.axvline(x=x_val, color='#e0d8c8', linewidth=0.3, linestyle=':', alpha=0.4)
    for y_val in np.arange(0, size_y + 0.1, 0.5):
        ax.axhline(y=y_val, color='#e0d8c8', linewidth=0.3, linestyle=':', alpha=0.4)

    for x_val in range(0, int(size_x) + 1, 2):
        ax.text(x_val, -0.45, f'{x_val}m', ha='center', va='top', fontsize=9,
                color='#666', fontweight='bold')
    for y_val in range(0, int(size_y) + 1, 2):
        ax.text(-0.55, y_val, f'{y_val}m', ha='right', va='center', fontsize=9,
                color='#666', fontweight='bold')

    comm_bg = FancyBboxPatch((-0.1, -0.1), 10.2, size_y + 0.2,
                              boxstyle="round,pad=0.05",
                              facecolor='#e8f0e8', edgecolor='#4a7c4a',
                              linewidth=2.5, alpha=0.25)
    ax.add_patch(comm_bg)
    ax.text(5.0, size_y + 0.15, 'COMMERCIAL AREA (x: 0~10m)',
            ha='center', va='bottom', fontsize=14, color='#2a5c2a', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#c8e8c8', alpha=0.7, edgecolor='#4a7c4a'))

    wh_bg = FancyBboxPatch((9.9, -0.1), 6.2, size_y + 0.2,
                            boxstyle="round,pad=0.05",
                            facecolor='#e8e0f0', edgecolor='#5a4a7c',
                            linewidth=2.5, alpha=0.25)
    ax.add_patch(wh_bg)
    ax.text(13.0, size_y + 0.15, 'WAREHOUSE AREA (x: 10~16m)',
            ha='center', va='bottom', fontsize=14, color='#3a2a5c', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#d8c8e8', alpha=0.7, edgecolor='#5a4a7c'))

    ax.axvline(x=10.0, color='#ff4444', linewidth=3, linestyle='--', alpha=0.7)

    wall_color = '#8B4513'
    wall_lw = 3.5
    ax.plot([0, size_x], [0, 0], color=wall_color, linewidth=wall_lw)
    ax.plot([0, size_x], [size_y, size_y], color=wall_color, linewidth=wall_lw)
    ax.plot([0, 0], [0, size_y], color=wall_color, linewidth=wall_lw)
    ax.plot([size_x, size_x], [0, size_y], color=wall_color, linewidth=wall_lw)

    comm_shelf_color = '#5a8ab5'
    comm_shelf_dark = '#2a4a6a'

    comm_active = ld['active_shelvings'][0]
    draw_shelf_rect(ax, comm_active['x'], comm_active['y'],
                    comm_active['l'], comm_active['w'],
                    comm_active['orientation'],
                    '#3a7aa5', '#1a4a6a',
                    'CS-0', f'({comm_active["x"]:.1f},{comm_active["y"]:.1f})',
                    fontsize=7, label_color='yellow')

    comm_idx = 0
    for s in ld['inactive_shelvings']:
        comm_idx += 1
        sid = f'CS-{comm_idx}'
        coord_str = f'({s["x"]:.1f},{s["y"]:.1f})'
        draw_shelf_rect(ax, s['x'], s['y'], s['l'], s['w'], s['orientation'],
                        comm_shelf_color, comm_shelf_dark,
                        sid, coord_str, fontsize=6, label_color='white')

        arr = load_arrangement(scene_dir, s['name'])
        prods = get_products_from_arrangement(arr)
        n_prod = len(prods)
        if prods:
            unique_prods = sorted(set(prods))
            prod_str = ', '.join(p.split('.')[-1][:8] for p in unique_prods[:3])
            if len(unique_prods) > 3:
                prod_str += f' +{len(unique_prods)-3}'
            ax.text(s['x'], s['y'] - s['w']/2 - 0.12, f'{n_prod}pcs: {prod_str}',
                    ha='center', va='top', fontsize=4, color='#2a4a6a', style='italic')
        else:
            ax.text(s['x'], s['y'] - s['w']/2 - 0.12, 'EMPTY',
                    ha='center', va='top', fontsize=5, color='#cc3333', fontweight='bold')

    for s in ld['inactive_wall_shelvings']:
        draw_shelf_rect(ax, s['x'], s['y'], s['l'], s['w'], s['orientation'],
                        '#4a9ab5', '#2a6a8a',
                        'FREEZER', f'({s["x"]:.1f},{s["y"]:.1f})',
                        fontsize=6, label_color='white')

    row_colors = {
        'row_A_drinks': ('#c0392b', '#922b21', '[A] DRINKS'),
        'row_B_food':   ('#27ae60', '#1e8449', '[B] FOOD'),
        'row_C_daily':  ('#2980b9', '#1f618d', '[C] DAILY'),
    }

    row_product_labels = {
        'row_A_drinks': ['Beer', 'Soda', 'Juice'],
        'row_B_food':   ['Coffee', 'Milk', 'Cereal'],
        'row_C_daily':  ['BodyCare', 'Detergent', 'Paper'],
    }

    warehouse_shelves = [s for s in ld['active_shelvings']
                         if s['name'].startswith('warehouse_shelf')]

    current_row = None
    for ws in warehouse_shelves:
        parts = ws['name'].split(':')
        row_col = parts[1]
        row_name = '_'.join(row_col.split('_')[:3])
        col_idx = int(row_col.split('col')[1])

        fill_color, edge_color, row_label = row_colors[row_name]
        row_letter = row_name.split('_')[1]
        short_id = f'{row_letter}-C{col_idx}'
        coord_str = f'({ws["x"]:.1f},{ws["y"]:.1f})'

        draw_shelf_rect(
            ax, ws['x'], ws['y'], ws['l'], ws['w'], ws['orientation'],
            fill_color, edge_color, short_id, coord_str,
            fontsize=6, label_color='white'
        )

        arr = load_arrangement(scene_dir, ws['name'])
        prods = get_products_from_arrangement(arr)
        n_prod = len(prods)
        if prods:
            unique_prods = sorted(set(prods))
            prod_str = ', '.join(p.split('.')[-1][:8] for p in unique_prods[:3])
            if len(unique_prods) > 3:
                prod_str += f' +{len(unique_prods)-3}'
            ax.text(ws['x'], ws['y'] + ws['w']/2 + 0.08,
                    f'{n_prod}pcs: {prod_str}',
                    ha='center', va='bottom', fontsize=3.5, color=edge_color)

        if row_name != current_row:
            current_row = row_name
            row_shelves = [s for s in warehouse_shelves if row_name in s['name']]
            if row_shelves:
                avg_y = np.mean([s['y'] for s in row_shelves])
                min_x = min(s['x'] - s['l']/2 for s in row_shelves)
                ax.text(min_x - 0.2, avg_y, row_label,
                        ha='right', va='center', fontsize=11, fontweight='bold',
                        color=fill_color,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  alpha=0.85, edgecolor=fill_color, linewidth=2))

                sub_labels = row_product_labels.get(row_name, [])
                for si, sl in enumerate(sub_labels):
                    ax.text(min_x - 0.2, avg_y - 0.4 - si * 0.28, sl,
                            ha='right', va='center', fontsize=6, color=edge_color)

    wh_rows = {}
    for ws in warehouse_shelves:
        parts = ws['name'].split(':')
        row_name = '_'.join(parts[1].split('_')[:3])
        if row_name not in wh_rows:
            wh_rows[row_name] = []
        wh_rows[row_name].append(ws)

    row_keys = sorted(wh_rows.keys())
    for i in range(len(row_keys) - 1):
        row_a = wh_rows[row_keys[i]]
        row_b = wh_rows[row_keys[i + 1]]
        y_a = max(s['y'] + s['w']/2 for s in row_a)
        y_b = min(s['y'] - s['w']/2 for s in row_b)
        ax.fill_between([10.3, 15.7], y_a, y_b, color='#f0e8d0', alpha=0.4)
        aisle_y = (y_a + y_b) / 2
        aisle_w = y_b - y_a
        ax.text(15.85, aisle_y, f'Aisle\n{aisle_w:.2f}m',
                ha='left', va='center', fontsize=7, color='#8a7a5a',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.7, edgecolor='#8a7a5a'))

    robot_x, robot_y = 11.0, 5.0
    draw_robot_arrow(ax, robot_x, robot_y, angle_deg=robot_angle, length=0.7)

    ax.text(5.0, 0.25, '2m Main Passage', ha='center', va='bottom', fontsize=9,
            color='#5a5a5a', style='italic',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.6))

    legend_elements = [
        patches.Patch(facecolor='#5a8ab5', edgecolor='#2a4a6a',
                       label='Commercial Shelf (CS-0 ~ CS-N)'),
        patches.Patch(facecolor='#3a7aa5', edgecolor='#1a4a6a',
                       label='Commercial Active Shelf (CS-0)'),
        patches.Patch(facecolor='#4a9ab5', edgecolor='#2a6a8a',
                       label='Wall Freezer'),
        patches.Patch(facecolor='#c0392b', edgecolor='#922b21',
                       label='Warehouse Row A -- Drinks (A-C0~C3)'),
        patches.Patch(facecolor='#27ae60', edgecolor='#1e8449',
                       label='Warehouse Row B -- Food (B-C0~C3)'),
        patches.Patch(facecolor='#2980b9', edgecolor='#1f618d',
                       label='Warehouse Row C -- Daily (C-C0~C3)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6600',
                   markeredgecolor='black', markersize=10,
                   label='Robot (arrow = heading)'),
        plt.Line2D([0], [0], color='#ff4444', linewidth=2.5, linestyle='--',
                   label='Zone Boundary (x=10m)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9,
              framealpha=0.9, edgecolor='#888', fancybox=True,
              title='LEGEND', title_fontsize=11)

    coord_info = (
        "COORDINATE SYSTEM\n"
        "============================\n"
        "Origin: bottom-left corner\n"
        "X axis: -> (0~16m)\n"
        "Y axis: ^  (0~10m)\n"
        "Unit:  meters\n"
        "============================\n"
        "Shelf ID format:\n"
        "  CS-N   = Commercial Shelf\n"
        "  A-C0~3 = Warehouse Drinks\n"
        "  B-C0~3 = Warehouse Food\n"
        "  C-C0~3 = Warehouse Daily\n"
        "============================\n"
        "Each shelf shows:\n"
        "  Line1: Shelf ID\n"
        "  Line2: (x, y) center coord\n"
        "============================\n"
        "Robot arrow = heading dir\n"
        "  0 deg = right (+X)\n"
        "  90 deg = up (+Y)"
    )
    ax.text(size_x + 0.5, size_y, coord_info,
            ha='left', va='top', fontsize=8, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                      alpha=0.95, edgecolor='#888'))

    shelf_coords = "SHELF COORDINATE TABLE\n"
    shelf_coords += "=" * 60 + "\n"
    shelf_coords += f"{'ID':<10} {'Center(X,Y)':<18} {'X Range':<18} {'Y Range':<18} {'Items'}\n"
    shelf_coords += "-" * 60 + "\n"

    shelf_coords += "--- Commercial ---\n"
    cs_idx = 0
    ca = comm_active
    shelf_coords += (f"{'CS-0':<10} ({ca['x']:6.2f}, {ca['y']:5.2f})   "
                     f"[{ca['x']-ca['l']/2:.2f}, {ca['x']+ca['l']/2:.2f}]   "
                     f"[{ca['y']-ca['w']/2:.2f}, {ca['y']+ca['w']/2:.2f}]   active\n")
    for s in ld['inactive_shelvings']:
        cs_idx += 1
        shelf_coords += (f"{'CS-'+str(cs_idx):<10} ({s['x']:6.2f}, {s['y']:5.2f})   "
                         f"[{s['x']-s['l']/2:.2f}, {s['x']+s['l']/2:.2f}]   "
                         f"[{s['y']-s['w']/2:.2f}, {s['y']+s['w']/2:.2f}]   "
                         f"{len(get_products_from_arrangement(load_arrangement(scene_dir, s['name'])))}\n")

    shelf_coords += "--- Warehouse ---\n"
    for ws in warehouse_shelves:
        parts = ws['name'].split(':')
        row_col = parts[1]
        row_name = '_'.join(row_col.split('_')[:3])
        col_idx = int(row_col.split('col')[1])
        short_id = f"{row_name.split('_')[1]}-C{col_idx}"
        arr = load_arrangement(scene_dir, ws['name'])
        prods = get_products_from_arrangement(arr)
        shelf_coords += (f"{short_id:<10} ({ws['x']:6.2f}, {ws['y']:5.2f})   "
                         f"[{ws['x']-ws['l']/2:.2f}, {ws['x']+ws['l']/2:.2f}]   "
                         f"[{ws['y']-ws['w']/2:.2f}, {ws['y']+ws['w']/2:.2f}]   "
                         f"{len(prods)}\n")

    ax.text(-1.0, -1.0, shelf_coords,
            ha='left', va='top', fontsize=5.5, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.9, edgecolor='#aaa'))

    ax.set_xticks(np.arange(0, size_x + 1, 1))
    ax.set_yticks(np.arange(0, size_y + 1, 1))
    ax.set_xticks(np.arange(0, size_x + 0.5, 0.5), minor=True)
    ax.set_yticks(np.arange(0, size_y + 0.5, 0.5), minor=True)
    ax.grid(which='major', alpha=0.3, linewidth=0.5)
    ax.grid(which='minor', alpha=0.15, linewidth=0.3)

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(scene_dir, 'top_down_map.png')
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f'Map saved to: {output_path}')

    json_output = output_path.replace('.png', '_coords.json')
    coord_data = {
        'room': {'size_x': size_x, 'size_y': size_y},
        'coordinate_system': {
            'origin': 'bottom-left corner',
            'x_axis': 'right (0~16m)',
            'y_axis': 'up (0~10m)',
            'unit': 'meters',
        },
        'zones': {
            'commercial': {'x_range': [0.0, 10.0], 'y_range': [0.0, 10.0]},
            'warehouse': {'x_range': [10.0, 16.0], 'y_range': [0.0, 10.0]},
        },
        'warehouse_zones': {
            'row_A_drinks': {
                'label': 'Drinks',
                'description': 'Beverages: beer, soda, juice',
                'shelf_ids': ['A-C0', 'A-C1', 'A-C2', 'A-C3'],
                'product_categories': ['BEER', 'DRINKS_SODA', 'JUICE'],
                'products': [
                    'food.BEER.DuffBeerCan',
                    'food.BEER.HeinekenLagerBeerBottle',
                    'food.DRINKS_SODA.Coca-ColaOriginal0.33L',
                    'food.DRINKS_SODA.FantaSaborNaranja2L',
                    'food.JUICE.MinuteMaidOrangeJuice',
                    'food.JUICE.TropicaOrangeJuice',
                ],
            },
            'row_B_food': {
                'label': 'Food',
                'description': 'Food: coffee, milk, cereal',
                'shelf_ids': ['B-C0', 'B-C1', 'B-C2', 'B-C3'],
                'product_categories': ['drinks.coffee', 'dairy_products', 'grocery'],
                'products': [
                    'food.drinks.coffeePackaging',
                    'food.drinks.coffeePackaging1',
                    'food.dairy_products.milkCarton',
                    'food.dairy_products.milkHandle',
                    'food.grocery.nestleFitnessChocolateCerealBox',
                    'food.grocery.cornFlakesRetroEditionSmall',
                ],
            },
            'row_C_daily': {
                'label': 'Daily Necessities',
                'description': 'Daily necessities: body care, detergent, household',
                'shelf_ids': ['C-C0', 'C-C1', 'C-C2', 'C-C3'],
                'product_categories': ['HYGIENE', 'HOUSEHOLD'],
                'products': [
                    'food.HYGIENE.NiveaBodyMilk',
                    'food.HYGIENE.NiveaBodyLotion',
                    'food.HOUSEHOLD.AceDetergent',
                    'food.HOUSEHOLD.TideDetergent',
                    'food.HOUSEHOLD.VanishStainRemover',
                    'food.HOUSEHOLD.AjaxDishSoap',
                ],
            },
        },
        'robot_start': {'x': robot_x, 'y': robot_y, 'heading_deg': robot_angle},
        'commercial_shelves': [],
        'warehouse_shelves': [],
    }

    ca = comm_active
    coord_data['commercial_shelves'].append({
        'id': 'CS-0',
        'name': ca['name'],
        'center': {'x': round(ca['x'], 3), 'y': round(ca['y'], 3)},
        'size': {'l': round(ca['l'], 3), 'w': round(ca['w'], 3)},
        'orientation': ca['orientation'],
        'x_range': [round(ca['x'] - ca['l']/2, 3), round(ca['x'] + ca['l']/2, 3)],
        'y_range': [round(ca['y'] - ca['w']/2, 3), round(ca['y'] + ca['w']/2, 3)],
        'product_count': len(get_products_from_arrangement(
            load_arrangement(scene_dir, ca['name']))),
        'products': sorted(set(get_products_from_arrangement(
            load_arrangement(scene_dir, ca['name'])))),
    })

    cs_idx = 0
    for s in ld['inactive_shelvings']:
        cs_idx += 1
        arr = load_arrangement(scene_dir, s['name'])
        prods = get_products_from_arrangement(arr)
        coord_data['commercial_shelves'].append({
            'id': f'CS-{cs_idx}',
            'name': s['name'],
            'center': {'x': round(s['x'], 3), 'y': round(s['y'], 3)},
            'size': {'l': round(s['l'], 3), 'w': round(s['w'], 3)},
            'orientation': s['orientation'],
            'x_range': [round(s['x'] - s['l']/2, 3), round(s['x'] + s['l']/2, 3)],
            'y_range': [round(s['y'] - s['w']/2, 3), round(s['y'] + s['w']/2, 3)],
            'product_count': len(prods),
            'products': sorted(set(prods)),
        })

    for ws in warehouse_shelves:
        parts = ws['name'].split(':')
        row_col = parts[1]
        row_name = '_'.join(row_col.split('_')[:3])
        col_idx = int(row_col.split('col')[1])
        short_id = f"{row_name.split('_')[1]}-C{col_idx}"
        arr = load_arrangement(scene_dir, ws['name'])
        prods = get_products_from_arrangement(arr)
        coord_data['warehouse_shelves'].append({
            'id': short_id,
            'name': ws['name'],
            'row': row_name,
            'col': col_idx,
            'center': {'x': round(ws['x'], 3), 'y': round(ws['y'], 3)},
            'size': {'l': round(ws['l'], 3), 'w': round(ws['w'], 3)},
            'orientation': ws['orientation'],
            'x_range': [round(ws['x'] - ws['l']/2, 3), round(ws['x'] + ws['l']/2, 3)],
            'y_range': [round(ws['y'] - ws['w']/2, 3), round(ws['y'] + ws['w']/2, 3)],
            'product_count': len(prods),
            'products': sorted(set(prods)),
        })

    with open(json_output, 'w') as f:
        json.dump(coord_data, f, indent=2, ensure_ascii=False)
    print(f'Coordinates saved to: {json_output}')

    return output_path, json_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate top-down map for restock scene')
    parser.add_argument('scene_dir', help='Path to scene directory')
    parser.add_argument('-o', '--output', default=None, help='Output image path')
    parser.add_argument('--robot-angle', type=float, default=0,
                        help='Robot heading angle in degrees (0=right, 90=up)')
    args = parser.parse_args()
    generate_map(args.scene_dir, args.output, args.robot_angle)
