# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025


export _wsave, set_plot_configs

_wsave(s, fig::Figure; dpi::Int = 300) = fig.savefig(s, bbox_inches = "tight", dpi = dpi)


function set_plot_configs(; fontsize = 10)
    set_style("whitegrid")
    rc("font", family = "serif", size = fontsize)
    font_prop = matplotlib.font_manager.FontProperties(
        family = "serif",
        style = "normal",
        size = fontsize,
    )
    sfmt = matplotlib.ticker.ScalarFormatter(useMathText = true)
    sfmt.set_powerlimits((0, 0))
    matplotlib.use("Agg")

    return font_prop, sfmt
end

