from db_query import fetch_latest_record
from visualization.latex_table import LatexTable, MultiColumn

datasets = ( 'kdd99', 'kdd99_dropped', 'creditcardfraud', 'ecoli', 'optical_digits', 'satimage', 'pen_digits', 'abalone', 'sick_euthyroid',
'spectrometer', 'car_eval_34', 'isolet', 'us_crime', 'yeast_ml8', 'scene', 'libras_move', 'thyroid_sick', 'coil_2000',
'arrhythmia', 'solar_flare_m0', 'oil', 'car_eval_4', 'wine_quality', 'letter_img', 'yeast_me2', 'webpage',
'ozone_level', 'mammography', 'protein_homo', 'abalone_19' )

# for debug
datasets = datasets[:3]

# latex用のため，`_`を`\_`に置き換え
datasets_ = [c.replace('_', r'\_') for c in datasets]

layers = (
    [0],
    [20, 10, 5],
    [20, 15, 10],
    [20, 15, 10, 5],
)
layers_str = (
    '[0]',
    '[20,10,5]',
    '[20,15,10]',
    '[20,15,10,5]',
)
models = (
    'Logistic Regression',
    'SVM',
    'Random Forest',
    'LightGBM',
    'Multi Perceptron'
)
mdls = ('lr', 'svm', 'rf', 'lgb', 'mp')
mdl_dict = {mdls[i]: models[i] for i in range(5)}
used_classes = ('all', 'minority', 'majority')

pps = ('なし', '正規化', '標準化')

class ResultTable(LatexTable):
    def __init__(
            self, 
            mdl: str,  # Model
            hpt: bool,  # Hyper Parameter Tuning
            pp: str = pps[0],  # Preprocess
            aepp: str = pps[0],  # AutoEncoder Preprocess
            aeclass: str = 'all'
            ):
        caption = f"{mdl_dict[mdl]} optuna:{hpt} 前処理:{pp} AE前処理:{aepp} AE 学習クラス:{aeclass}"
        label = f"{mdl}+op={hpt}+pp={pp}+aepp={aepp}+ae={aeclass}"
        if aepp == pps[0]:
            col_n = 8
            layers_col = ["layers", 'none', *layers_str[1:4], 'none', *layers_str[1:4]]
        else:
            col_n = 6
            layers_col = ["layers", *layers_str[1:4], *layers_str[1:4]]
        super().__init__(
            caption=caption,
            label=label,
            format=f"p{{22mm}}*{col_n // 2}{'{p{14mm}}|'}*{col_n // 2}{'{p{14mm}}'}",
            column_num=col_n + 1,
        )
        self.add_hline()
        self.add_columns(["optuna", MultiColumn(hpt, col_n)])
        self.add_columns(["前処理", MultiColumn(pp, col_n)])
        self.add_columns(["AE前処理", MultiColumn(aepp, col_n)])
        self.add_hline()

        # align right
        for i in range(len(layers_col)):
            if i == 0:
                continue
            layers_col[i] = MultiColumn(layers_col[i], 1, 'r')
        self.add_columns(layers_col)
        self.add_hline()
        self.add_columns(
            ['Dataset', MultiColumn('minority F-accuracy', col_n // 2),
                        MultiColumn('macro F-accuracy', col_n // 2),])
        self.add_hline()


        self.mdl = mdl
        self.hpt = hpt
        self.pp = pp
        self.aepp = aepp
        self.aeclass = aeclass

        self.col_n = col_n

    def fetch_results(self):
        for i in range(len(datasets)):
            dataset = datasets[i]
            cols = [datasets_[i]] # escaped underscore
            res_minority = []
            res_macro = []
            if self.pp == pps[0]:
                normalize = False
                standardize = False
            elif self.pp == pps[1]:
                normalize = True
                standardize = False
            elif self.pp == pps[2]:
                normalize = False
                standardize = True
            else:
                raise ValueError
            if self.aepp == pps[0]:
                ae_normalize = False
                ae_standardize = False
            elif self.aepp == pps[1]:
                ae_normalize = True
                ae_standardize = False
            elif self.aepp == pps[2]:
                ae_normalize = False
                ae_standardize = True
            else:
                raise ValueError

            for layer in layers:
                if layer == [0] and self.col_n == 6:
                    continue
                record = fetch_latest_record({
                    'dataset.name': dataset,
                    'dataset.standardization': standardize,
                    'dataset.normalization': normalize,
                    'model.name': self.mdl,
                    'model.optuna': self.hpt,
                    'ae.layers': layer,
                    'ae.used_class': self.aeclass,
                    'ae.standardization': ae_standardize,
                    'ae.normalization': ae_normalize,
                    },
                    projection={
                        '_id': 0,
                        "result.macro.f1": 1,
                        "result.minority.f1": 1,
                    })
                if record is None:
                    res_minority.append('-')
                    res_macro.append('-')
                else:
                    res_minority.append(f"{record['result']['minority']['f1']:.3f}")
                    res_macro.append(f"{record['result']['macro']['f1']:.3f}")
            
            # 最大値を太字にする,ただし0の場合は太字にしない
            max_minority = max(res_minority)
            max_macro = max(res_macro)
            min_minority = min(res_minority)
            min_macro = min(res_macro)

            for i in range(len(res_minority)):
                if res_minority[i] == max_minority != min_minority:
                    res_minority[i] = self._bf(res_minority[i])
                if res_macro[i] == max_macro != min_macro:
                    res_macro[i] = self._bf(res_macro[i])
                # align right
                res_minority[i] = MultiColumn(res_minority[i], 1, 'r')
                res_macro[i] = MultiColumn(res_macro[i], 1, 'r')
            
            cols.extend(res_minority)
            cols.extend(res_macro)
            self.add_columns(cols)

if __name__ == '__main__':
    table = ResultTable('lr', False, 'false', 'false', 'all')
    table.fetch_results()
    with open(f"thesis/tables/{table.label}.tex", "w") as f:
        f.write(table.compile())

    # for mdl in mdls:
    #     for hpt in [True, False]:
    #         for pp in pps:
    #             for aepp in pps:
    #                 for aeclass in used_classes:
    #                     table = ResultTable(mdl, hpt, pp, aepp, aeclass)
    #                     table.fetch_results()
    #                     with open(f"results/{table.label}.tex", "w") as f:
    #                         f.write(str(table))

            