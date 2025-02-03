from src.plot.plotting import *

def plot_everything():
    all_experiments = extract_data()

    plot_unsemantic_results(all_experiments)
    plot_ablations_without_regret(all_experiments)
    plot_main_results(all_experiments)
    plot_main_results_with_phi_and_approximate(all_experiments)
    plot_scaling_qwen_results_compact(all_experiments)
    plot_naive_plus_temperature_sensitivity(all_experiments)
    plot_scaling_qwen_results(all_experiments)
    plot_scaling_qwen_results_compact_context_efficiency(all_experiments)
    plot_scaling_qwen_results_stability(all_experiments)
    plot_naive_plus_temperature_sensitivity(all_experiments)
    plot_p_keep_search_results(all_experiments)
    plot_context_length_and_strategy_comparison(all_experiments)
    plot_approximate_beam_sizes_results(all_experiments)
    plot_approximate_detailed_results(all_experiments)
    plot_approximate_context_sampling_method_comparison(all_experiments)

if __name__ == "__main__":
    plot_everything()