"""全ての実験プログラムのエントリポイント
このプログラムは、原則変更禁止。
非破壊的変更を加える際は、パッチバージョンを上げること。

Supported BaseFlow: 1.4.0
"""
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from itertools import product
from tkinter import Y

from _main import flows, runners, load_config


def main():
    cfg = load_config()
    with ProcessPoolExecutor(max_workers=2) as executor:
        for model, dataset, layers \
                in product(cfg['models'], cfg['datasets'], cfg['layers']):
            print(dataset, model, layers)
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
            future.add_done_callback(clean_up)


def clean_up(return_value):
    del return_value


if __name__ == '__main__':
    main()
