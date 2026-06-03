import json
import random
import sys
import hashlib
import argparse
import numpy as np
from pathlib import Path
from dataclasses import asdict
from omegaconf import OmegaConf

sys.path.append('.')
from dsynth.scene_gen.hydra_configs import ShelfConfig, FillingType, ShelfType
from dsynth.scene_gen.scene_generator import product_filling_from_shelf_config
from dsynth.scene_gen.layouts.layout_generator import TensorFieldLayout
from dsynth.assets.asset import load_assets_lib
from dsynth.scene_gen.utils import flatten_dict
from dsynth.scene_gen.arrangements import set_shelf, add_objects_to_shelf_v2
from dsynth.assets.ss_assets import DefaultShelf
import scene_synthesizer as synth

SIZE_X = 16.0
SIZE_Y = 10.0
COMMERCIAL_X = 10.0
OUTPUT_DIR = 'generated_envs/restock_scene'

SHELF_TWO_SIDED_L = 1.074
SHELF_TWO_SIDED_W = 0.827
SHELF_TWO_SIDED_BOARDS = 3

WAREHOUSE_ROWS = 3
WAREHOUSE_COLS = 4
WAREHOUSE_SHELF_GAP_X = 0.3
WAREHOUSE_ROW_GAP_Y = 1.5
WAREHOUSE_START_X = 10.0
WAREHOUSE_START_Y = 1.5

R1_WAREHOUSE_ROWS = 2
R1_WAREHOUSE_COLS = 3
R1_WAREHOUSE_BOARDS = 2
R1_WAREHOUSE_START_Y = 2.5

ROW_PRODUCTS = {
    'row_A_drinks': {
        0: {'food.BEER.DuffBeerCan': 8, 'food.BEER.HeinekenLagerBeerBottle': 6},
        1: {'food.DRINKS_SODA.FantaSaborNaranja2L': 6, 'food.DRINKS_SODA.Coca-ColaOriginal0.33L': 8},
        2: {'food.JUICE.TropicaOrangeJuice': 6, 'food.JUICE.MinuteMaidOrangeJuice': 6},
    },
    'row_B_food': {
        0: {'food.drinks.coffeePackaging': 8, 'food.drinks.coffeePackaging1': 6},
        1: {'food.dairy_products.milkCarton': 8, 'food.dairy_products.milkHandle': 6},
        2: {'food.grocery.nestleFitnessChocolateCerealBox': 6, 'food.grocery.cornFlakesRetroEditionSmall': 6},
    },
    'row_C_daily': {
        0: {'food.HYGIENE.NiveaBodyMilk': 8, 'food.HYGIENE.NiveaBodyLotion': 6},
        1: {'food.HOUSEHOLD.AceDetergent': 6, 'food.HOUSEHOLD.TideDetergent': 6},
        2: {'food.HOUSEHOLD.VanishStainRemover': 6, 'food.HOUSEHOLD.AjaxDishSoap': 6},
    },
}

R1_ROW_PRODUCTS = {
    'row_A_drinks': {
        1: {'food.DRINKS_SODA.FantaSaborNaranja2L': 4, 'food.DRINKS_SODA.Coca-ColaOriginal0.33L': 4},
        2: {'food.BEER.DuffBeerCan': 4, 'food.JUICE.TropicaOrangeJuice': 4},
    },
    'row_B_daily': {
        1: {'food.HYGIENE.NiveaBodyMilk': 4, 'food.HYGIENE.NiveaBodyLotion': 4},
        2: {'food.HOUSEHOLD.VanishStainRemover': 4, 'food.HOUSEHOLD.AjaxDishSoap': 4},
    },
}


def load_product_assets_lib():
    assets_cfg = OmegaConf.load('conf/assets/assets_preprocessed.yaml')
    assets_cfg = OmegaConf.merge({'assets': assets_cfg}, {'assets': {'assets_dir': 'dsynth/assets'}})
    return flatten_dict(load_assets_lib(assets_cfg.assets, disable_caching=True), sep='.'), assets_cfg


def generate_commercial_layout(product_assets_lib, assets_cfg, seed=42, robot_type='fetch'):
    if robot_type == 'r1':
        inactive_shelvings_list = ['fixtures.shelf_metal']
        inactive_wall_shelvings_list = []
        skip_prob = 0.5
    else:
        inactive_shelvings_list = ['fixtures.shelf_metal']
        inactive_wall_shelvings_list = ['fixtures.freezer_large']
        skip_prob = 0.1

    cfg = OmegaConf.create({
        'ds_continuous': {
            'name': 'commercial',
            'size_x': COMMERCIAL_X,
            'size_y': SIZE_Y,
            'max_tries': 20,
            'tf_blending_decay': 12.0,
            'inactive_wall_shelvings_occupancy_width': 0.4,
            'inactive_shelvings_occupancy_width': 1.0,
            'inactive_shelvings_skip_prob': skip_prob,
            'fixtures_occupancy_width': 0.2,
            'passage_width': 2.0,
            'scene_fixtures_list': [],
            'active_wall_shelvings_list': [],
            'inactive_wall_shelvings_list': inactive_wall_shelvings_list,
            'inactive_shelvings_list': inactive_shelvings_list,
            'active_shelvings_list': [],
            'output_dir': '/tmp',
        },
        'assets': assets_cfg.assets,
    })

    rng = random.Random(seed)
    layout_gen = TensorFieldLayout(name='commercial', cfg=cfg, rng=rng)
    layout_data = layout_gen()

    if layout_data is None:
        print("WARNING: Commercial layout generation failed, using fallback")
        layout_data = {
            'service': [{'name': 'blocked_area', 'x': COMMERCIAL_X, 'y': SIZE_Y - 2, 'l': 2, 'w': 2, 'orientation': 'horizontal', 'occupancy_width': 0.0, 'asset_name': None}],
            'scene_fixtures': [],
            'inactive_wall_shelvings': [],
            'active_wall_shelvings': [],
            'inactive_shelvings': [],
            'active_shelvings': [],
        }

    return layout_data


def build_warehouse_shelves(robot_type='fetch'):
    shelves = []
    if robot_type == 'r1':
        row_names = ['row_A_drinks', 'row_B_daily']
        n_rows = R1_WAREHOUSE_ROWS
        n_cols = R1_WAREHOUSE_COLS
        start_y = R1_WAREHOUSE_START_Y
    else:
        row_names = ['row_A_drinks', 'row_B_food', 'row_C_daily']
        n_rows = WAREHOUSE_ROWS
        n_cols = WAREHOUSE_COLS
        start_y = WAREHOUSE_START_Y

    for row_idx in range(n_rows):
        row_y = start_y + row_idx * (SHELF_TWO_SIDED_W + WAREHOUSE_ROW_GAP_Y)
        for col_idx in range(n_cols):
            shelf_x = WAREHOUSE_START_X + (SHELF_TWO_SIDED_L / 2 + 0.3) + col_idx * (SHELF_TWO_SIDED_L + WAREHOUSE_SHELF_GAP_X)
            row_name = row_names[row_idx]
            shelf_name = f'warehouse_shelf:{row_name}_col{col_idx}'
            shelves.append({
                'name': shelf_name,
                'x': shelf_x,
                'y': row_y,
                'l': SHELF_TWO_SIDED_L,
                'w': SHELF_TWO_SIDED_W,
                'orientation': 'horizontal',
                'occupancy_width': 0.3,
                'asset_name': 'fixtures.small_shelf_two_sided',
                'row_name': row_name,
                'row_idx': row_idx,
                'col_idx': col_idx,
            })
    return shelves


def generate_warehouse_arrangement(shelf_name, row_name, product_assets_lib, rng, robot_type='fetch'):
    if robot_type == 'r1':
        board_product_numcol = R1_ROW_PRODUCTS[row_name]
        n_boards = SHELF_TWO_SIDED_BOARDS
        start_board = 1
        end_board = SHELF_TWO_SIDED_BOARDS
        num_products_per_block = 4
        num_products_per_board = 6
    else:
        board_product_numcol = ROW_PRODUCTS[row_name]
        n_boards = SHELF_TWO_SIDED_BOARDS
        start_board = 0
        end_board = SHELF_TWO_SIDED_BOARDS
        num_products_per_block = 7
        num_products_per_board = 10

    shelf_config = ShelfConfig(
        name=shelf_name,
        filling_type=FillingType.BOARDWISE_COLUMNS,
        queries=[],
        num_products_per_block=num_products_per_block,
        num_products_per_board=num_products_per_board,
        start_filling_board=start_board,
        end_filling_from_board=end_board,
        is_dynamic=True,
        num_boards=n_boards,
        x_gap=0.03,
        y_gap=0.03,
        delta_x=0.0,
        delta_y=0.0,
        start_point_x=-1.0,
        start_point_y=-1.0,
        noise_std_x=0.0,
        noise_std_y=0.0,
        rotation_lower=0.0,
        rotation_upper=0.0,
        shelf_asset='fixtures.small_shelf_two_sided',
        shelf_type=ShelfType.SHELF,
        shuffle_boards=False,
        shuffle_items_on_board=True,
        board_product_numcol=board_product_numcol,
    )

    cfg_dict = OmegaConf.create(asdict(shelf_config))
    filling, _, _ = product_filling_from_shelf_config(cfg_dict, list(product_assets_lib.keys()), rng=rng)

    scene = synth.Scene()
    shelf = product_assets_lib['fixtures.small_shelf_two_sided'].ss_asset

    support_data = set_shelf(
        scene, shelf, 0, 0, 0,
        f'SHELF_{shelf_name}',
        f'support_SHELF_{shelf_name}',
    )

    add_objects_to_shelf_v2(
        scene, 0, filling, product_assets_lib, support_data,
        cfg_dict.x_gap, cfg_dict.y_gap,
        cfg_dict.delta_x, cfg_dict.delta_y,
        cfg_dict.start_point_x, cfg_dict.start_point_y,
        cfg_dict.filling_type, 42,
        cfg_dict.noise_std_x, cfg_dict.noise_std_y,
        cfg_dict.rotation_lower, cfg_dict.rotation_upper,
    )

    json_str = synth.exchange.export.export_json(scene, include_metadata=False)
    data = json.loads(json_str)
    del data["geometry"]
    return data


def generate_commercial_arrangement(product_assets_lib, rng, robot_type='fetch'):
    if robot_type == 'r1':
        num_products_per_block = 4
        num_products_per_board = 6
        num_boards = 3
        start_filling_board = 0
        end_filling_from_board = 3
        board_product_numcol = {
            1: {'food.HYGIENE.NiveaBodyMilk': 3, 'food.drinks.coffeePackaging': 3},
            2: {'food.BEER.DuffBeerCan': 3, 'food.DRINKS_SODA.FantaSaborNaranja2L': 3},
        }
    else:
        num_products_per_block = 7
        num_products_per_board = 10
        num_boards = 5
        start_filling_board = 0
        end_filling_from_board = 5
        board_product_numcol = {
            1: {'food.HYGIENE.NiveaBodyMilk': 4, 'food.drinks.coffeePackaging': 4},
            2: {'food.grocery.nestleFitnessChocolateCerealBox': 3, 'food.dairy_products.milkCarton': 4},
            3: {'food.BEER.DuffBeerCan': 5, 'food.DRINKS_SODA.FantaSaborNaranja2L': 4},
        }

    shelf_config = ShelfConfig(
        name='commercial_shelf',
        filling_type=FillingType.BOARDWISE_COLUMNS,
        queries=[],
        num_products_per_block=num_products_per_block,
        num_products_per_board=num_products_per_board,
        start_filling_board=start_filling_board,
        end_filling_from_board=end_filling_from_board,
        is_dynamic=True,
        num_boards=num_boards,
        x_gap=0.05,
        y_gap=0.05,
        delta_x=0.0,
        delta_y=0.0,
        start_point_x=-1.0,
        start_point_y=-1.0,
        noise_std_x=0.0,
        noise_std_y=0.0,
        rotation_lower=0.0,
        rotation_upper=0.0,
        shelf_asset='fixtures.shelf_metal',
        shelf_type=ShelfType.SHELF,
        shuffle_boards=True,
        shuffle_items_on_board=True,
        board_product_numcol=board_product_numcol,
    )

    cfg_dict = OmegaConf.create(asdict(shelf_config))
    filling, _, _ = product_filling_from_shelf_config(cfg_dict, list(product_assets_lib.keys()), rng=rng)

    scene = synth.Scene()
    shelf = product_assets_lib['fixtures.shelf_metal'].ss_asset

    support_data = set_shelf(
        scene, shelf, 0, 0, 0,
        'SHELF_commercial_shelf',
        'support_SHELF_commercial_shelf',
    )

    add_objects_to_shelf_v2(
        scene, 0, filling, product_assets_lib, support_data,
        cfg_dict.x_gap, cfg_dict.y_gap,
        cfg_dict.delta_x, cfg_dict.delta_y,
        cfg_dict.start_point_x, cfg_dict.start_point_y,
        cfg_dict.filling_type, 42,
        cfg_dict.noise_std_x, cfg_dict.noise_std_y,
        cfg_dict.rotation_lower, cfg_dict.rotation_upper,
    )

    json_str = synth.exchange.export.export_json(scene, include_metadata=False)
    data = json.loads(json_str)
    del data["geometry"]
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--robot', type=str, default='fetch', choices=['fetch', 'r1'],
                        help='Robot type: fetch (default) or r1 (R1Lite)')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory (default: generated_envs/restock_scene[_r1])')
    args = parser.parse_args()

    robot_type = args.robot
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        'generated_envs/restock_scene_r1' if robot_type == 'r1' else 'generated_envs/restock_scene'
    )
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    product_assets_lib, assets_cfg = load_product_assets_lib()
    rng = random.Random(42)

    # === Step 1: Generate commercial area layout using TensorField ===
    print("[1/4] Generating commercial area layout (TensorField)...")
    commercial_layout = generate_commercial_layout(product_assets_lib, assets_cfg, robot_type=robot_type)

    # === Step 2: Build warehouse shelves with fixed positions ===
    print("[2/4] Building warehouse shelves (fixed layout)...")
    warehouse_shelves = build_warehouse_shelves(robot_type=robot_type)

    # === Step 3: Merge layouts ===
    print("[3/4] Merging commercial + warehouse layouts...")

    # Add blocked area for warehouse zone to prevent commercial shelves from overlapping
    commercial_layout['service'].append({
        'name': 'warehouse_zone_blocked',
        'x': COMMERCIAL_X + 3.0,
        'y': SIZE_Y / 2,
        'l': SIZE_Y,
        'w': SIZE_X - COMMERCIAL_X,
        'orientation': 'vertical',
        'occupancy_width': 0.0,
        'asset_name': None,
    })

    # Convert warehouse shelves to layout format - as active_shelvings with fixed positions
    warehouse_active_shelvings = []
    for ws in warehouse_shelves:
        warehouse_active_shelvings.append({
            'name': ws['name'],
            'x': ws['x'],
            'y': ws['y'],
            'l': ws['l'],
            'w': ws['w'],
            'orientation': ws['orientation'],
            'occupancy_width': ws['occupancy_width'],
            'asset_name': ws['asset_name'],
        })

    # Add active shelving for commercial area (x=0,y=0 means it will be replaced from inactive)
    commercial_active = {
        'name': 'commercial_shelf',
        'x': 0,
        'y': 0,
        'l': 1.554,
        'w': 0.554,
        'orientation': 'horizontal',
        'occupancy_width': 0.2,
        'asset_name': 'fixtures.shelf_metal',
    }

    # Merge all layout data
    merged_layout = {
        'service': commercial_layout['service'],
        'scene_fixtures': commercial_layout['scene_fixtures'],
        'inactive_wall_shelvings': commercial_layout['inactive_wall_shelvings'],
        'active_wall_shelvings': commercial_layout['active_wall_shelvings'],
        'inactive_shelvings': commercial_layout['inactive_shelvings'],
        'active_shelvings': [commercial_active] + warehouse_active_shelvings,
    }

    # Build fake_arrangements_mapping - warehouse shelves get their own arrangement files
    fake_arrangements_mapping = {
        'fixtures.shelf': ['fixtures.shelf_fake_1', 'fixtures.shelf_fake_2', 'fixtures.shelf_fake_3'],
        'fixtures.shelf_metal': ['fixtures.shelf_metal_fake_1', 'fixtures.shelf_metal_fake_2'],
        'fixtures.small_shelf_two_sided': ['fixtures.small_shelf_two_sided_fake_1', 'fixtures.small_shelf_two_sided_fake_2'],
    }
    # Add warehouse shelf arrangement mappings (not needed since they're active_shelvings now)

    config_hash = hashlib.sha1(str(42).encode()).hexdigest()[-8:]
    scene_name = f'restock_scene_{config_hash}_0'

    result = {
        'layout_data': merged_layout,
        'size_x': SIZE_X,
        'size_y': SIZE_Y,
        'fake_arrangements_mapping': fake_arrangements_mapping,
        'hash': config_hash,
        'scene_id': 0,
        'scene_name': scene_name,
    }

    layout_path = output_dir / f'layout_data_{scene_name}.json'
    with open(layout_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f'  Saved: {layout_path}')

    # === Step 4: Generate arrangement files ===
    print("[4/4] Generating product arrangements...")

    # Commercial area active shelf arrangement
    commercial_arr = generate_commercial_arrangement(product_assets_lib, rng, robot_type=robot_type)
    commercial_arr_path = output_dir / 'commercial_shelf.json'
    with open(commercial_arr_path, 'w') as f:
        json.dump(commercial_arr, f, indent=4)
    print(f'  Commercial shelf: {commercial_arr_path}')

    # Warehouse shelf arrangements
    total_warehouse_products = 0
    for ws in warehouse_shelves:
        arr_data = generate_warehouse_arrangement(ws['name'], ws['row_name'], product_assets_lib, rng, robot_type=robot_type)
        arr_path = output_dir / f'{ws["name"]}.json'
        with open(arr_path, 'w') as f:
            json.dump(arr_data, f, indent=4)
        num_products = sum(1 for node in arr_data.get('graph', []) if '/' not in node[1] and 'SHELF' not in node[1])
        total_warehouse_products += num_products
    print(f'  Warehouse: {len(warehouse_shelves)} shelves, {total_warehouse_products} products')

    # Save input_config.yaml - resolve all interpolations first
    assets_cfg_resolved = OmegaConf.to_container(assets_cfg, resolve=True)
    input_config = OmegaConf.create({
        'ds_continuous': {
            'name': 'restock_scene',
            'size_x': SIZE_X,
            'size_y': SIZE_Y,
            'active_shelvings_list': [{
                'name': 'commercial_shelf',
                'filling_type': 'BOARDWISE_COLUMNS',
                'queries': [],
                'num_products_per_block': 7,
                'num_products_per_board': 10,
                'start_filling_board': 0,
                'end_filling_from_board': 5,
                'is_dynamic': True,
                'num_boards': 5,
                'x_gap': 0.05,
                'y_gap': 0.05,
                'delta_x': 0.0,
                'delta_y': 0.0,
                'start_point_x': -1.0,
                'start_point_y': -1.0,
                'noise_std_x': 0.0,
                'noise_std_y': 0.0,
                'rotation_lower': 0.0,
                'rotation_upper': 0.0,
                'shelf_asset': 'fixtures.shelf_metal',
                'shelf_type': 'SHELF',
                'shuffle_boards': True,
                'shuffle_items_on_board': True,
                'board_product_numcol': {
                    1: {'food.HYGIENE.NiveaBodyMilk': 4, 'food.drinks.coffeePackaging': 4},
                    2: {'food.grocery.nestleFitnessChocolateCerealBox': 3, 'food.dairy_products.milkCarton': 4},
                    3: {'food.BEER.DuffBeerCan': 5, 'food.DRINKS_SODA.FantaSaborNaranja2L': 4},
                },
            }],
            'active_wall_shelvings_list': [],
            'inactive_shelvings_list': ['fixtures.shelf_metal', 'fixtures.small_shelf_two_sided'],
            'inactive_wall_shelvings_list': ['fixtures.freezer_large'],
            'scene_fixtures_list': [],
            'fake_arrangements_mapping': fake_arrangements_mapping,
            'num_scenes': 1,
            'num_workers': 1,
            'output_dir': str(output_dir),
            'rewrite': True,
            'show': False,
            'layout_gen_type': 'PROCEDURAL_TENSOR_FIELD',
            'randomize_layout': True,
            'randomize_arrangements': True,
            'random_seed': 42,
            'max_tries': 20,
            'tf_blending_decay': 12.0,
            'inactive_wall_shelvings_occupancy_width': 0.4,
            'inactive_shelvings_occupancy_width': 1.0,
            'inactive_shelvings_skip_prob': 0.1,
            'fixtures_occupancy_width': 0.2,
            'passage_width': 2.0,
        },
        'assets': assets_cfg_resolved['assets'],
    })
    with open(output_dir / 'input_config.yaml', 'w') as f:
        f.write(OmegaConf.to_yaml(input_config))

    print(f'\n=== Scene generation complete! ===')
    print(f'Robot type: {robot_type}')
    print(f'Output: {output_dir}')
    print(f'Commercial area: {len(commercial_layout["inactive_shelvings"])} shelves (random)')
    n_rows = R1_WAREHOUSE_ROWS if robot_type == 'r1' else WAREHOUSE_ROWS
    n_cols = R1_WAREHOUSE_COLS if robot_type == 'r1' else WAREHOUSE_COLS
    n_boards = R1_WAREHOUSE_BOARDS if robot_type == 'r1' else SHELF_TWO_SIDED_BOARDS
    print(f'Warehouse area: {len(warehouse_shelves)} shelves (fixed, {n_rows} rows x {n_cols} cols, {n_boards} boards)')
    print(f'\nRun:')
    robot_flag = ' -r ds_r1' if robot_type == 'r1' else ''
    print(f'  python scripts/show_env_in_sim.py {output_dir}/ --sim-backend cpu --render-backend cpu')
    print(f'  python scripts/run_keyboard_control.py {output_dir}/ --sim-backend cpu --render-backend cpu{robot_flag}')


if __name__ == '__main__':
    main()
