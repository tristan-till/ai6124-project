import numpy as np

from utils.aggregation import AggregationLayer

import utils.plot as plot
import utils.params as params

import utils.evo_data as evo_data

from utils.backbone import get_backbone
from utils.baseline import get_baseline

def test_aggregation(device, target=params.TARGET, model_path=params.BEST_MODEL_PATH, baseline_path=params.BL_BEST_MODEL_PATH, plot_path=params.AGG_PLOT):
    base, prices = evo_data.supplementary_data(target)
    model_preds = evo_data.get_predictions(device, model_path=model_path, model_loader=get_backbone)
    baseline_preds = evo_data.get_predictions(device, model_path=baseline_path, model_loader=get_baseline)
    _, model_data, _, model_prices = evo_data.stack_and_trim(model_preds, prices, base, device)
    _, baseline_data, _, _ = evo_data.stack_and_trim(baseline_preds, prices, base, device)
    zero_change, random, perfect_preds, trivial_preds = evo_data.get_trivial_predictions(base, model_prices, device)
    _, num_inputs = model_data.shape
    portfolios, buys, sells = [], [], []
    agg = AggregationLayer(num_inputs=num_inputs, device=device)
    for preds in [model_data, baseline_data, zero_change, random, perfect_preds, trivial_preds]:
        for inp, price in zip(preds, model_prices):
            agg.decide(inp, price)
        p, b, s = agg.manager.get_history()
        portfolios.append(p)
        buys.append(b)
        sells.append(s)
        agg.manager.reset()
    plot.plot_generation(model_prices, portfolios, buys, sells, plot_path)

if __name__ == '__main__':
    test_aggregation('cuda')