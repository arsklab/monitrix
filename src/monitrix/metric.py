from dataclasses import dataclass, field
from typing import Literal, Any
from itertools import groupby

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
from itertools import product, starmap


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure

import pandas as pd
import numpy as np


@dataclass
class metric:

    true: pd.Series = field(repr=False)
    pred: pd.Series = field(repr=False)

    conf: pd.Series = field(repr=False)

    key: Any = None

    @classmethod
    def c_matrix(cls, true_series: pd.Series, pred_series: pd.Series) -> pd.DataFrame:

        _true_series = true_series.fillna("na").astype(str)
        _pred_series = pred_series.fillna("na").astype(str)

        _enc = OrdinalEncoder()
        _enc.fit(pd.concat([_true_series, _pred_series]).values.reshape(-1, 1))
        categories = _enc.categories_[0]

        y_true = _enc.transform(_true_series.values.reshape(-1, 1))
        pred_y = _enc.transform(_pred_series.values.reshape(-1, 1))

        cm = pd.DataFrame(
            confusion_matrix(y_true, pred_y), index=categories, columns=categories
        )
        cm.index.name = "True"
        cm.columns.name = "Pred"
        return cm

    def confusion_matrix(self, name: str | None = None) -> pd.DataFrame:
        cm = self.c_matrix(self.true, self.pred)
        cm = cm.loc[(cm != 0).any(axis=1), :]
        cm.name = name
        return cm

    @property
    def true_str(self) -> pd.Series:
        return self.true.fillna("na").astype(str)

    @property
    def pred_str(self) -> pd.Series:
        return self.pred.fillna("na").astype(str)

    @property
    def pred_conf(self) -> pd.Series:
        return self.conf

    @property
    def true_positives(self) -> list[int]:
        return self.true_str[
            (self.true_str == self.pred_str) & (~self.pred.isna())
        ].index.tolist()

    @property
    def false_positives(self) -> list[int]:
        return self.true_str[
            (self.true_str != self.pred_str) & (~self.pred.isna())
        ].index.tolist()

    @property
    def true_negatives(self) -> list[int]:
        return self.true_str[
            (self.true_str == self.pred_str) & self.pred.isna()
        ].index.tolist()

    @property
    def false_negatives(self) -> list[int]:
        return self.true_str[
            (self.true_str != self.pred_str) & self.pred.isna()
        ].index.tolist()

    @property
    def mistakes(self) -> list[int]:
        return self.false_positives + self.false_negatives

    @property
    def tp(self) -> int:
        return len(self.true_positives)

    @property
    def fp(self) -> int:
        return len(self.false_positives)

    @property
    def tn(self) -> int:
        return len(self.true_negatives)

    @property
    def fn(self) -> int:
        return len(self.false_negatives)

    def __len__(self) -> int:
        return len(self.true)

    def accuracy(
        self,
        average: Literal["weighted", "macro", "micro"] = "micro",
        zero_division: Literal[0, 1] = 0,
    ) -> float:
        """average,zero_divisionはインターフェース共有化のために定義してあり、機能はしない"""
        return (self.tp + self.tn) / len(self)

    def precision(
        self,
        average: Literal["weighted", "macro", "micro"] = "micro",
        zero_division: Literal[0, 1] = 0,
    ) -> float:
        match average:
            case "micro":
                try:
                    return self.tp / (self.tp + self.fp)
                except ZeroDivisionError:
                    return zero_division
            case _:
                return float(
                    precision_score(
                        self.true_str,
                        self.pred_str,
                        average=average,
                        zero_division=zero_division,
                    )
                )

    def recall(
        self,
        average: Literal["weighted", "macro", "micro"] = "micro",
        zero_division: Literal[0, 1] = 0,
    ) -> float:
        match average:
            case "micro":
                try:
                    return self.tp / (self.tp + self.fn)
                except ZeroDivisionError:
                    return zero_division
            case _:
                return float(
                    recall_score(
                        self.true_str,
                        self.pred_str,
                        average=average,
                        zero_division=zero_division,
                    )
                )

    def f1(
        self,
        average: Literal["weighted", "macro", "micro"] = "micro",
        zero_division: Literal[0, 1] = 0,
    ) -> float:
        match average:
            case "micro":
                try:
                    return 2 * self.tp / (2 * self.tp + self.fp + self.fn)
                except ZeroDivisionError:
                    return zero_division
            case _:
                return float(
                    f1_score(
                        self.true_str,
                        self.pred_str,
                        average=average,
                        zero_division=zero_division,
                    )
                )

    def scores(self, zero_division: Literal[0, 1] = 0) -> dict[str, float | int]:
        _scores = {
            "total samples": len(self),
            "accuracy": self.accuracy(),
            "cm_tp": self.tp,
            "cm_tn": self.tn,
            "cm_fp": self.fp,
            "cm_fn": self.fn,
        }
        for s, a in product(
            ["precision", "recall", "f1"], ["weighted", "macro", "micro"]
        ):
            _scores[f"{s}_{a}"] = getattr(self, s)(a, zero_division)
        return _scores


class metrics(list["metric"]):
    def getattr(self, name: str, *arg, **kwargs) -> list[float]:
        return [
            (
                getattr(m, name)(*arg, **kwargs)
                if callable(getattr(m, name))
                else getattr(m, name)
            )
            for m in self
        ]

    def average(
        self, method: Literal["weighted", "macro", "micro"], name: str, *arg, **kwargs
    ) -> float:
        match method:
            case "weighted":  # サンプル数で重み付けした平均
                return sum(
                    starmap(
                        lambda a, b: a * b,
                        zip(self.getattr(name, *arg, **kwargs), (len(m) for m in self)),
                    )
                ) / sum(len(m) for m in self)
            case "macro":  # マクロ平均（各パラメータを等しく重み付け）
                return float(np.mean(self.getattr(name, *arg, **kwargs)))
            case "micro":
                return getattr(self, name)(*arg, **kwargs)

    @property
    def tp(self) -> int:
        return int(sum(self.getattr("tp")))

    @property
    def fp(self) -> int:
        return int(sum(self.getattr("fp")))

    @property
    def tn(self) -> int:
        return int(sum(self.getattr("tn")))

    @property
    def fn(self) -> int:
        return int(sum(self.getattr("fn")))

    def accuracy(self, zero_division: Literal[0, 1] = 0) -> float:
        return (self.tp + self.tn) / sum(len(m) for m in self)

    def precision(self, zero_division: Literal[0, 1] = 0) -> float:
        try:
            return self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            return zero_division

    def recall(self, zero_division: Literal[0, 1] = 0) -> float:
        try:
            return self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            return zero_division

    def f1(self, zero_division: Literal[0, 1] = 0) -> float:
        try:
            return 2 * self.tp / (2 * self.tp + self.fp + self.fn)
        except ZeroDivisionError:
            return zero_division

    def scores(self, zero_division: Literal[0, 1] = 0) -> dict[str, float | int]:
        total_samples = sum(len(m) for m in self)

        _scores = {
            "total samples": total_samples,
            "cm_tp": self.tp,
            "cm_tn": self.tn,
            "cm_fp": self.fp,
            "cm_fn": self.fn,
        }
        for s, a in product(
            ["accuracy", "precision", "recall", "f1"], ["weighted", "macro", "micro"]
        ):
            _scores[f"{s}_{a}"] = self.average(a, s, zero_division=zero_division)
        return _scores


class mdataframe(pd.DataFrame):

    def add_true(
        self, true_dataframe: pd.DataFrame, re_monitor_id: dict[int, int] | None = None
    ):
        for require in ["frame", "monitor_id", "class", "true_value"]:
            if not require in true_dataframe.columns:
                raise ValueError

        _df = self.copy()
        if re_monitor_id is not None:
            _df["pred_monitor_id"] = _df["monitor_id"].copy()
            _df["monitor_id"] = _df["monitor_id"].replace(re_monitor_id)
        _df = _df[~_df["is_monitor"]]
        # _df = _df.drop(["is_monitor"],axis=1)

        return mdataframe(
            pd.merge(
                true_dataframe, _df, on=["frame", "monitor_id", "class"], how="left"
            )
        )

    def metrics(self, group: str | list[str] | None = None) -> metrics:
        if not "true_value" in self.keys():
            raise ValueError

        if group is None:
            return metrics(
                [
                    metric(
                        self["true_value"],
                        self["pred_value"],
                        self["object_conf"] * self["ocr_conf"],
                    )
                ]
            )
        return metrics(
            [
                metric(
                    data["true_value"],
                    data["pred_value"],
                    data["object_conf"] * data["ocr_conf"],
                    key=key,
                )
                for key, data in self.groupby(group, dropna=False)
            ]
        )

    def timeplot(
        self,
        figsize: tuple[int, int] = (12, 8),
        class_names: dict[int, str] | None = None,
        class_units: dict[int, str] | None = None,
        scores: dict[str, tuple[str, str | None]] | None = None,
        value_line_width: float = 1,
        conf_line_width: float = 0.5,
        conf_line_alpha: float = 0.2,
        conf_area_alpha: float = 0.05,
    ) -> list[Figure]:

        figs = []

        x_min, x_max = self["frame"].min(), self["frame"].max()
        no_monitor_mask = self["is_monitor"].fillna(False)
        if (scores is not None) and ("true_value" in self.keys()):
            _metrics = {m.key: m for m in self.metrics(["monitor_id", "class"])}
        for _id, monitor in self[~no_monitor_mask].groupby(
            "monitor_id", dropna=True
        ):  # , dropna=False):
            classes = sorted(monitor["class"].unique())
            fig, line_axs = plt.subplots(
                len(classes), 1, sharex=True, figsize=figsize, squeeze=False
            )
            for ax, cls in zip(line_axs.flatten(), classes):
                as2 = ax.twinx()
                ax.set_xlim(x_min, x_max)

                values = monitor[monitor["class"] == cls]
                # 欠損部分をnanで埋める
                values = values.set_index("frame", drop=True).sort_index()
                values = values.reindex(range(x_min, x_max + 1)).reset_index()

                if class_names is not None:
                    if sub_title := class_names.get(cls, None):
                        ax.set_title(sub_title, fontsize="medium", loc="left")
                if class_units is not None:
                    if sub_ylabel := class_units.get(cls, None):
                        ax.set_ylabel(sub_ylabel)

                if "pred_monitor_id" in monitor.keys():
                    for pid, pred_monitor in values.groupby(
                        "pred_monitor_id", dropna=True
                    ):
                        pred_monitor = pred_monitor.set_index(
                            "frame", drop=True
                        ).sort_index()
                        pred_monitor = pred_monitor.reindex(
                            range(
                                pred_monitor.index.min(), pred_monitor.index.max() + 1
                            )
                        ).reset_index()
                        ax.plot(
                            pred_monitor["frame"],
                            pred_monitor["pred_value"],
                            lw=value_line_width,
                        )

                        area = pred_monitor["object_conf"] * pred_monitor["ocr_conf"]
                        as2.plot(
                            pred_monitor["frame"],
                            area,
                            lw=conf_line_width,
                            alpha=conf_line_alpha,
                        )
                        as2.fill_between(
                            pred_monitor["frame"], 0, area, alpha=conf_area_alpha
                        )
                else:
                    ax.plot(values["frame"], values["pred_value"], lw=value_line_width)

                    area = values["object_conf"] * values["ocr_conf"]
                    as2.plot(
                        values["frame"], area, lw=conf_line_width, alpha=conf_line_alpha
                    )
                    as2.fill_between(values["frame"], 0, area, alpha=conf_area_alpha)

                ax.minorticks_on()
                ax.grid(which="both", zorder=0, color="lightgray", alpha=0.5)
                as2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
                as2.set_ylim(0, 1)

                if "true_value" in monitor.keys():
                    mask = values["pred_value"] != values["true_value"]
                    p_miss = values["pred_value"][mask]
                    ax.scatter(
                        values["frame"][mask],
                        p_miss,
                        marker="x",
                        s=20,
                        c="red",
                        alpha=0.8,
                        linewidths=0.5,
                        zorder=0,
                    )
                    mask = mask & values["pred_value"].isna()
                    n_miss = values["true_value"][mask]
                    ax.scatter(
                        values["frame"][mask],
                        n_miss,
                        marker="*",
                        s=20,
                        c="gray",
                        alpha=0.8,
                        linewidths=0.5,
                        zorder=0,
                    )

                    if scores is not None:
                        if _metric := _metrics.get((int(_id), int(cls))):
                            _score = _metric.scores()
                            _legends = []
                            for k, v in scores.items():
                                if k in _score.keys():
                                    _value = _score[k]
                                    _label = v[0]
                                    if fmt := v[1]:
                                        _legends.append(f"{_label}{_value:{fmt}}")
                                    else:
                                        _legends.append(f"{_label}{_value}")
                            ax.legend(["\n".join(_legends)], handlelength=0)

            fig.supxlabel("frame")
            fig.supylabel("conf area", x=1.0)
            fig.suptitle(f"id={int(_id)}")
            fig.tight_layout()
            figs.append(fig)
        return figs

    def metrics_summary(
        self,
        zero_division: Literal[0, 1] = 0,
        group: str | list[str] | None = None,
    ) -> pd.DataFrame:

        if isinstance(group, list) and (len(group) == 1):
            group = group[0]
        _ms = self.metrics(group)
        key: tuple[str, ...] | str | None = _ms[0].key

        if isinstance(key, tuple):
            current_level = len(key)
            cols = group
            key_f = lambda _level: lambda x: x.key[:_level]
            cols_f = lambda _keys: dict(zip(cols, _keys))
        elif isinstance(key, (str, int, float)):
            current_level = 1
            cols = [group]
            key_f = lambda _level: lambda x: x.key
            cols_f = lambda _key: dict(zip(cols, [_key]))
        elif key is None:
            current_level = 0
            cols = []
            key_f = lambda _level: None
            cols_f = lambda _key: dict()
        else:
            print(key)
            raise ValueError

        _values = []
        if isinstance(key, tuple) or (key is None):
            while current_level > -1:
                for k, v in groupby(_ms, key=key_f(current_level)):
                    _values.append(
                        {
                            "level": current_level,
                            **cols_f(k),
                            **metrics(v).scores(zero_division),
                        }
                    )
                current_level -= 1
        else:
            for k, v in groupby(_ms, key=key_f(current_level)):
                _values.append(
                    {
                        "level": current_level,
                        **cols_f(k),
                        **metrics(v).scores(zero_division),
                    }
                )
            _values.append({"level": 0, **_ms.scores(zero_division)})

        return pd.DataFrame(_values).set_index(["level"] + cols).sort_index()


def cheak(true_data: pd.DataFrame) -> bool:
    """正解データのチェック"""
    required_columns = {"frame", "monitor_id", "class", "true_value"}
    assert required_columns.issubset(set(true_data.columns))

    for col in required_columns:
        # 数値データかどうか
        assert pd.api.types.is_numeric_dtype(true_data[col])
    return True
