"""全ての実験プログラムのエントリポイント
このプログラムは、原則変更禁止。
非破壊的変更を加える際は、パッチバージョンを上げること。

Supported BaseFlow: 1.2.4
"""
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from itertools import product
from multiprocessing import cpu_count

from loguru import logger

from _main import flows, runners, load_config


def main():
    cfg = load_config()
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures: dict[str, Future] = dict()
        for layers, model, dataset \
                in product(cfg['layers'], cfg['models'], cfg['datasets']):
            params = cfg['default'].copy()
            params['encoder_param']['layers'] = layers
            Flow = flows[dataset['name']]
            for k, v in dataset.items():
                if k == 'name':
                    pass
                else:
                    params[k] = v

            runner = runners[model]

            future = runner(params, executor, Flow)
            futures[f"{model}{layers}{dataset['name']}"] = future
        for k in futures:
            #
            try:
                logger.info(f"{k} is done: {futures[k].result()}")
            except Exception as e:
                logger.error(f"task {k} raised error: {e}")


if __name__ == '__main__':
    main()
