import sys
import inspect
from datetime import datetime
from pathlib import Path
import shutil
from .log import *

import mxnet as mx


from lib.utils.cmd_args import get_train_arguments, get_test_arguments


def init_exp(run_file_path, add_exp_args):
    if 'train' in sys.argv:
        parser = get_train_arguments()
    else:
        parser = get_test_arguments()
    parser = add_exp_args(parser)

    args = parser.parse_args()
    stdout_log_path = None

    if args.mode == 'train':
        run_file_path = Path(run_file_path)
        exp_path = run_file_path.parent

        run_name = args.mode + datetime.strftime(datetime.today(), '_%Y-%m-%d_%H-%M-%S')
        run_path = exp_path / 'runs' / run_name
        args.logs_path = run_path / 'logs'
        args.run_path = run_path
        args.checkpoints_path = run_path / 'checkpoints'

        if not args.no_exp:
            assert not run_path.exists()
            run_path.mkdir(parents=True)
            shutil.copy(str(run_file_path), str(run_path / 'run.py'))

            if not args.checkpoints_path.exists():
                args.checkpoints_path.mkdir(parents=True)
            if not args.logs_path.exists():
                args.logs_path.mkdir(parents=True)

            stdout_log_path = args.logs_path / 'train_log.txt'
    else:
        run_path = Path(args.run_path)
        args.logs_path = run_path / 'logs'

        current_date = datetime.strftime(datetime.today(), '%Y-%m-%d_%H-%M-%S')
        stdout_log_path = args.logs_path / f'test_log_{current_date}.txt'

        if args.vizualization:
            test_viz_path = args.logs_path / f'viz_{current_date}'
            if not test_viz_path.exists():
                test_viz_path.mkdir()
            args.viz_path = test_viz_path

        run_weights = sorted(run_path.rglob('*.params'), key=lambda x: x.stem)
        assert run_weights, "Can't find model weights"

        args.weights = str(run_weights[-1])

    if stdout_log_path is not None:
        fh = logging.FileHandler(str(stdout_log_path))
        formatter = logging.Formatter(fmt='(%(levelname)s) %(asctime)s: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if args.no_cuda:
        logger.info('Using CPU')
        args.kvstore = 'local'
        args.ctx = mx.cpu(0)
    else:
        if args.gpus:
            args.ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
            args.ngpus = len(args.ctx)
        else:
            args.ctx = [mx.gpu(i) for i in range(args.ngpus)]
        logger.info(f'Number of GPUs: {args.ngpus}')

        if args.ngpus < 2:
            args.syncbn = False

    logger.info(args)

    # Print test function source
    if args.mode == 'test':
        run_module = load_python_file_as_module(run_file_path)

        logger.info('\nSource code of the test function:')
        logger.info(inspect.getsource(run_module.test) + '\n')

    return args


def load_python_file_as_module(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module