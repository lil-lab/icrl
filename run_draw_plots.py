from src.plot.plotting import *

def plot_everything():
    all_experiments = extract_data()

    print("Plotting main results")
    plot_main_results(all_experiments)
    print("Plotting ablations")
    plot_ablations(all_experiments)
    print("Plotting ablations without regret")
    plot_ablations_without_regret(all_experiments)
    print("Plotting approximate beam sizes results without regret")
    plot_approximate_beam_sizes_results_without_regret(all_experiments)
    print("Plotting p_keep search results")
    plot_p_keep_search_results(all_experiments)
    print("Plotting context length and strategy comparison")
    plot_context_length_and_strategy_comparison(all_experiments)
    print("Plotting approximate beam sizes results")
    plot_approximate_beam_sizes_results(all_experiments)
    print("Plotting compact naive heatmap")
    plot_compact_naive_heatmap(all_experiments)
    print("Plotting approximate detailed results")
    plot_approximate_detailed_results(all_experiments)
    print("Plotting approximate sampling comparisons")
    plot_approximate_context_sampling_method_comparison(all_experiments)
    print("Plotting naive plus vs explorative regret and accuracy")
    plot_naive_plus_vs_explorative_regret_and_accuracy(all_experiments)

if __name__ == "__main__":
    plot_everything()