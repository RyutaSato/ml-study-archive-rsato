"""全ての実験プログラムのエントリポイント
このプログラムは、原則変更禁止。
非破壊的変更を加える際は、パッチバージョンを上げること。

Supported BaseFlow: 1.2.0
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import cpu_count

from loguru import logger

from _main import flows, runners, load_config

def main():
    cfg = load_config()
    with ProcessPoolExecutor() as executor:
        futures = dict()
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
            logger.info(f"{k} is done: {futures[k].result()}")


if __name__ == '__main__':
    main()