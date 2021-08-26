import collections
import json

import pandas as pd
import seaborn as sns
import torch

class Warehouse:
    def __init__(self):
        self.series = {}

    def log_metric(self, metric, values, tags={}):
        key = self.series_key(metric, tags)
        assert isinstance(values, collections.Mapping)
        assert isinstance(tags, collections.Mapping)
        if not key in self.series:
            self.series[key] = {
                "metric": metric,
                "tags": tags,
                "data": []
            }
        self.series[key]["data"].append(self.preprocess(values))

    def clear(self, metric, tags={}):
        keys = list(self.series.keys())
        for key in keys:
            series = self.series[key]
            if metric is not None and metric != series["metric"]:
                continue
            elif any(series["tags"].get(t, None) != v for t, v in tags.items()):
                continue
            else:
                del self.series[key]


    def query(self, metrics, tags={}):
        if not isinstance(metrics, list):
            metrics = [metrics]
        rows = []
        for serie in self.series.values():
            if not serie["metric"] in metrics:
                continue
            if not all(serie["tags"].get(t, None) == v for t, v in tags.items()):
                continue
            for entry in serie["data"]:
                rows.append({
                    "metric": serie["metric"],
                    **serie["tags"],
                    **entry,
                })
        return pd.DataFrame(rows)

    def show(self, metrics, tags={}, **kwargs):
        if not isinstance(metrics, list):
            metrics = [metrics]
        df = self.query(metrics, tags)
        g = sns.FacetGrid(df, **kwargs)
        g.map(sns.lineplot, "step", "value")
        if len(metrics) == 1:
            g.set_ylabels(metrics[0])
        g.set_xlabels("steps")
        return g

    @staticmethod
    def series_key(metric, tags={}):
        keys = sorted(set(tags.keys()))
        key_string = json.dumps({k: tags[k] for k in keys})
        if key_string != "":
            return f"{metric} {key_string}"
        else:
            return f"{metric}"

    @staticmethod
    def preprocess(value_dict):
        d = {}
        for k, v in value_dict.items():
            if torch.is_tensor(v):
                v = v.item()
            d[k] = v
        return d
