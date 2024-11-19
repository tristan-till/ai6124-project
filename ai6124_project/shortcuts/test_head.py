from ai6124_project.classes.manager import PortfolioManager

import ai6124_project.utils.plot as plot
import ai6124_project.utils.params as params
import ai6124_project.utils.evo_data as evo_data
import ai6124_project.utils.eval as evals
import ai6124_project.utils.helpers as helpers

from ai6124_project.utils.backbone import get_backbone

def test_head(device, in_test, p_test, genome_path=params.BEST_GENOME_PATH, plot_path=params.EVO_PLOT):
    _, num_inputs = in_test.shape
    portfolios, buys, sells = [], [], []
    manager = PortfolioManager(num_inputs=num_inputs, device=device, rule_operator=lambda x: sum(x) / len(x))
    manager.fis.load_genome(genome_path)
    for inp, price in zip(in_test, p_test):
        manager.forward(inp, price)
    p, b, s = manager.get_history()
    portfolios.append(p)
    buys.append(b)
    sells.append(s)
    plot.plot_generation(p_test, portfolios, buys, sells, plot_path)
    evals.head_benchmarks(in_test, p_test, p, device)
    manager.reset()

if __name__ == '__main__':
    device='cuda'
    helpers.set_deterministic()
    base, prices = evo_data.supplementary_data(params.TARGET)
    model_preds = evo_data.get_predictions(device, model_path=params.BEST_MODEL_PATH, model_loader=get_backbone, target=params.TARGET, supps=params.SUPP)
    _, model_data, _, model_prices = evo_data.stack_and_trim(model_preds, prices, base, device)
    ft_base, ft_prices = evo_data.supplementary_data(params.FT_TARGET)
    ft_model_preds = evo_data.get_predictions(device, model_path=params.FT_BEST_MODEL_PATH, model_loader=get_backbone, target=params.FT_TARGET, supps=params.FT_SUPP)
    _, ft_model_data, _, ft_model_prices = evo_data.stack_and_trim(ft_model_preds, ft_prices, ft_base, device)
    test_head(device, model_data, model_prices, genome_path=params.BEST_CR_GENOME_PATH)
    test_head(device, ft_model_data, ft_model_prices, genome_path=params.BEST_CR_GENOME_PATH, plot_path=params.FT_EVO_PLOT)
