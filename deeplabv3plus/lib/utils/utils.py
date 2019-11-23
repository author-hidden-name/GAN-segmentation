import mxnet as mx
from lib.utils.log import logger, TqdmToLogger


def save_checkpoint(net, args, epoch=None):
    if epoch is None:
        checkpoint_name = 'last_checkpoint.params'
    else:
        checkpoint_name = f'{epoch:03d}.params'

    if not args.checkpoints_path.exists():
        args.checkpoints_path.mkdir(parents=True)

    checkpoint_path = args.checkpoints_path / checkpoint_name
    logger.info(f'Save checkpoint to {str(checkpoint_path)}')
    net.save_parameters(str(checkpoint_path))


def split_and_load(inputs, ctx_list, batch_axis=0, even_split=False):
    r"""Split with support for kwargs dictionary"""
    def split_map(obj):
        if isinstance(obj, mx.ndarray.NDArray):
            return mx.gluon.utils.split_and_load(obj, ctx_list, batch_axis, even_split=even_split)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(split_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(split_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(split_map, obj.items()))))
        return [obj for _ in ctx_list]

    inputs = split_map(inputs) if len(inputs) > 0 else []
    inputs = tuple(inputs)

    return inputs

def get_ade20k_names():
    classes = ("wall", "building, edifice", "sky", "floor, flooring", "tree",
               "ceiling", "road, route", "bed", "windowpane, window", "grass",
               "cabinet", "sidewalk, pavement",
               "person, individual, someone, somebody, mortal, soul",
               "earth, ground", "door, double door", "table", "mountain, mount",
               "plant, flora, plant life", "curtain, drape, drapery, mantle, pall",
               "chair", "car, auto, automobile, machine, motorcar",
               "water", "painting, picture", "sofa, couch, lounge", "shelf",
               "house", "sea", "mirror", "rug, carpet, carpeting", "field", "armchair",
               "seat", "fence, fencing", "desk", "rock, stone", "wardrobe, closet, press",
               "lamp", "bathtub, bathing tub, bath, tub", "railing, rail", "cushion",
               "base, pedestal, stand", "box", "column, pillar", "signboard, sign",
               "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink",
               "skyscraper", "fireplace, hearth, open fireplace", "refrigerator, icebox",
               "grandstand, covered stand", "path", "stairs, steps", "runway",
               "case, display case, showcase, vitrine",
               "pool table, billiard table, snooker table", "pillow",
               "screen door, screen", "stairway, staircase", "river", "bridge, span",
               "bookcase", "blind, screen", "coffee table, cocktail table",
               "toilet, can, commode, crapper, pot, potty, stool, throne",
               "flower", "book", "hill", "bench", "countertop",
               "stove, kitchen stove, range, kitchen range, cooking stove",
               "palm, palm tree", "kitchen island",
               "computer, computing machine, computing device, data processor, "
               "electronic computer, information processing system",
               "swivel chair", "boat", "bar", "arcade machine",
               "hovel, hut, hutch, shack, shanty",
               "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, "
               "motorcoach, omnibus, passenger vehicle",
               "towel", "light, light source", "truck, motortruck", "tower",
               "chandelier, pendant, pendent", "awning, sunshade, sunblind",
               "streetlight, street lamp", "booth, cubicle, stall, kiosk",
               "television receiver, television, television set, tv, tv set, idiot "
               "box, boob tube, telly, goggle box",
               "airplane, aeroplane, plane", "dirt track",
               "apparel, wearing apparel, dress, clothes",
               "pole", "land, ground, soil",
               "bannister, banister, balustrade, balusters, handrail",
               "escalator, moving staircase, moving stairway",
               "ottoman, pouf, pouffe, puff, hassock",
               "bottle", "buffet, counter, sideboard",
               "poster, posting, placard, notice, bill, card",
               "stage", "van", "ship", "fountain",
               "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
               "canopy", "washer, automatic washer, washing machine",
               "plaything, toy", "swimming pool, swimming bath, natatorium",
               "stool", "barrel, cask", "basket, handbasket", "waterfall, falls",
               "tent, collapsible shelter", "bag", "minibike, motorbike", "cradle",
               "oven", "ball", "food, solid food", "step, stair", "tank, storage tank",
               "trade name, brand name, brand, marque", "microwave, microwave oven",
               "pot, flowerpot", "animal, animate being, beast, brute, creature, fauna",
               "bicycle, bike, wheel, cycle", "lake",
               "dishwasher, dish washer, dishwashing machine",
               "screen, silver screen, projection screen",
               "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase",
               "traffic light, traffic signal, stoplight", "tray",
               "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, "
               "dustbin, trash barrel, trash bin",
               "fan", "pier, wharf, wharfage, dock", "crt screen",
               "plate", "monitor, monitoring device", "bulletin board, notice board",
               "shower", "radiator", "glass, drinking glass", "clock", "flag")
    return classes