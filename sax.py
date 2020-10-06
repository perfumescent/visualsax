"""Implements PAA."""
from collections import defaultdict

import numpy as np
import colorsys
import random
import pandas as pd
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from pyecharts.options import *

from pyecharts.charts import Line
from pyecharts.globals import ThemeType


def znorm(series, znorm_threshold=0.01):
    """Znorm implementation."""
    sd = np.std(series)
    if sd < znorm_threshold:
        return series
    mean = np.mean(series)
    return (series - mean) / sd


def paa(series, segment_amount):
    """
    PAA implementation.
    @series list; original time series
    @segment_amount int; discretization size 'm'
    """
    series_len = len(series)

    # check for the trivial case
    if series_len == segment_amount:
        return np.copy(series)
    else:
        all_segments_center = np.zeros(segment_amount)  # 1-d array: single point value of each segment
        segments_x = np.zeros(segment_amount)
        # check when we are even
        if series_len % segment_amount == 0:  # if series is divisible

            segment_len = series_len // segment_amount
            for i in range(0, series_len):
                idx = i // segment_len
                np.add.at(all_segments_center, idx, series[i])
                # res[idx] = res[idx] + series[i]
            all_segments_center = all_segments_center / segment_len

            for i in range(0, segment_amount):
                segments_x[i] = segment_len * (i + 1)

            return segments_x, all_segments_center, np.linspace(0, series_len - 1, series_len), \
                   all_segments_center.repeat(segment_len)
        # and process when we are odd
        else:

            for i in range(0, segment_amount * series_len):
                idx = i // series_len
                pos = i // segment_amount
                np.add.at(all_segments_center, idx, series[pos])
                # res[idx] = res[idx] + series[pos]
            all_segments_center = all_segments_center / series_len

            for i in range(0, segment_amount):
                segments_x[i] += series_len * (i + 1) / segment_amount

            return segments_x, all_segments_center, np.linspace(0, series_len - 1, series_len * segment_amount), \
                   all_segments_center.repeat(series_len)


def sax(series, x, y, breakpoints):
    """
    Converts a normlized PAA timeseries to SAX symbols.
    @series np.array: the point of PAA series
    @x np.array: xaxis of full curve after PAA 
    @y np.array: yaxis of full curve after PAA 
    @breakpoints 1xα np.array:  [-np.inf, α-1 breakpoints]
    @:return a string separated by blank space; sax string e.g. "abbcda"

    """
    alpha = len(breakpoints)
    sax_list = list()
    sax_string_plot = dict()

    sax_plot = dict()
    for i in range(0, alpha):
        sax_plot[convert_to_letter(i)] = np.full(len(y), None, dtype=float)
        sax_string_plot[convert_to_letter(i)] = np.full(len(y), None, dtype=float)
        # 1*len(y) nan np.array

    for i in range(0, len(series)):

        ts_length = int(len(y) / len(series))
        idx = i * ts_length  # idx_letter_head; len(y) / len(series) is the length of original ts
        # if series[i] is below 0, start from the bottom, or else from the top
        if series[i] >= 0:
            j = alpha - 1
            while (j > 0) and (breakpoints[j] >= series[i]):
                j = j - 1
            # print(convert_to_letter(j))
            # print(sax_plot[convert_to_letter(j)])
            # print(idx)
            # print(idx + ts_length)
            sax_list.append(convert_to_letter(j))
            sax_plot[convert_to_letter(j)][idx:idx + ts_length - 1] = y[idx:idx + ts_length - 1]
            sax_string_plot[convert_to_letter(j)][idx:idx + ts_length - 1] = j
            # "-1" aims to remove the vertical line segment of the sax plot
        else:
            j = 1
            while j < alpha and breakpoints[j] <= series[i]:
                j = j + 1
            sax_list.append(convert_to_letter(j - 1))
            sax_string_plot[convert_to_letter(j - 1)][idx:idx + ts_length - 1] = j - 1
            sax_plot[convert_to_letter(j - 1)][idx:idx + ts_length - 1] = y[idx:idx + ts_length - 1]

    return ''.join(sax_list), x, sax_plot, sax_string_plot


def convert_to_letter(idx):
    """Convert a numerical index to a char by ASCII."""
    if 0 <= idx < 20:
        return chr(97 + idx)
    else:
        raise ValueError('A wrong idx value supplied.')


def is_mindist_zero(a, b):
    """Check mindist."""
    if len(a) != len(b):
        return 0
    else:
        for i in range(0, len(b)):
            if abs(ord(a[i]) - ord(b[i])) > 1:
                return 0
    return 1


def breakpoints(a_size):
    """Generate a set of alphabet cuts for its size."""
    """ Typically, we generate cuts in R as follows:
        get_cuts_for_num <- function(num) {
        cuts = c(-Inf)
        for (i in 1:(num-1)) {
            cuts = c(cuts, qnorm(i * 1/num))
            }
            cuts
        }

        get_cuts_for_num(3) """
    options = {
        2: np.array([-np.inf, 0.00]),
        3: np.array([-np.inf, -0.4307273, 0.4307273]),
        4: np.array([-np.inf, -0.6744898, 0, 0.6744898]),
        5: np.array([-np.inf, -0.841621233572914, -0.2533471031358,
                     0.2533471031358, 0.841621233572914]),
        6: np.array([-np.inf, -0.967421566101701, -0.430727299295457, 0,
                     0.430727299295457, 0.967421566101701]),
        7: np.array([-np.inf, -1.06757052387814, -0.565948821932863,
                     -0.180012369792705, 0.180012369792705, 0.565948821932863,
                     1.06757052387814]),
        8: np.array([-np.inf, -1.15034938037601, -0.674489750196082,
                     -0.318639363964375, 0, 0.318639363964375,
                     0.674489750196082, 1.15034938037601]),
        9: np.array([-np.inf, -1.22064034884735, -0.764709673786387,
                     -0.430727299295457, -0.139710298881862, 0.139710298881862,
                     0.430727299295457, 0.764709673786387, 1.22064034884735]),
        10: np.array([-np.inf, -1.2815515655446, -0.841621233572914,
                      -0.524400512708041, -0.2533471031358, 0, 0.2533471031358,
                      0.524400512708041, 0.841621233572914, 1.2815515655446]),
        11: np.array([-np.inf, -1.33517773611894, -0.908457868537385,
                      -0.604585346583237, -0.348755695517045,
                      -0.114185294321428, 0.114185294321428, 0.348755695517045,
                      0.604585346583237, 0.908457868537385, 1.33517773611894]),
        12: np.array([-np.inf, -1.38299412710064, -0.967421566101701,
                      -0.674489750196082, -0.430727299295457,
                      -0.210428394247925, 0, 0.210428394247925,
                      0.430727299295457, 0.674489750196082, 0.967421566101701,
                      1.38299412710064]),
        13: np.array([-np.inf, -1.42607687227285, -1.0200762327862,
                      -0.736315917376129, -0.502402223373355,
                      -0.293381232121193, -0.0965586152896391,
                      0.0965586152896394, 0.293381232121194, 0.502402223373355,
                      0.73631591737613, 1.0200762327862, 1.42607687227285]),
        14: np.array([-np.inf, -1.46523379268552, -1.06757052387814,
                      -0.791638607743375, -0.565948821932863, -0.36610635680057,
                      -0.180012369792705, 0, 0.180012369792705,
                      0.36610635680057, 0.565948821932863, 0.791638607743375,
                      1.06757052387814, 1.46523379268552]),
        15: np.array([-np.inf, -1.50108594604402, -1.11077161663679,
                      -0.841621233572914, -0.622925723210088,
                      -0.430727299295457, -0.2533471031358, -0.0836517339071291,
                      0.0836517339071291, 0.2533471031358, 0.430727299295457,
                      0.622925723210088, 0.841621233572914, 1.11077161663679,
                      1.50108594604402]),
        16: np.array([-np.inf, -1.53412054435255, -1.15034938037601,
                      -0.887146559018876, -0.674489750196082,
                      -0.488776411114669, -0.318639363964375,
                      -0.157310684610171, 0, 0.157310684610171,
                      0.318639363964375, 0.488776411114669, 0.674489750196082,
                      0.887146559018876, 1.15034938037601, 1.53412054435255]),
        17: np.array([-np.inf, -1.5647264713618, -1.18683143275582,
                      -0.928899491647271, -0.721522283982343,
                      -0.541395085129088, -0.377391943828554,
                      -0.223007830940367, -0.0737912738082727,
                      0.0737912738082727, 0.223007830940367, 0.377391943828554,
                      0.541395085129088, 0.721522283982343, 0.928899491647271,
                      1.18683143275582, 1.5647264713618]),
        18: np.array([-np.inf, -1.59321881802305, -1.22064034884735,
                      -0.967421566101701, -0.764709673786387,
                      -0.589455797849779, -0.430727299295457,
                      -0.282216147062508, -0.139710298881862, 0,
                      0.139710298881862, 0.282216147062508, 0.430727299295457,
                      0.589455797849779, 0.764709673786387, 0.967421566101701,
                      1.22064034884735, 1.59321881802305]),
        19: np.array([-np.inf, -1.61985625863827, -1.25211952026522,
                      -1.00314796766253, -0.8045963803603, -0.633640000779701,
                      -0.47950565333095, -0.336038140371823, -0.199201324789267,
                      -0.0660118123758407, 0.0660118123758406,
                      0.199201324789267, 0.336038140371823, 0.47950565333095,
                      0.633640000779701, 0.8045963803603, 1.00314796766253,
                      1.25211952026522, 1.61985625863827]),
        20: np.array([-np.inf, -1.64485362695147, -1.2815515655446,
                      -1.03643338949379, -0.841621233572914, -0.674489750196082,
                      -0.524400512708041, -0.385320466407568, -0.2533471031358,
                      -0.125661346855074, 0, 0.125661346855074, 0.2533471031358,
                      0.385320466407568, 0.524400512708041, 0.674489750196082,
                      0.841621233572914, 1.03643338949379, 1.2815515655446,
                      1.64485362695147]),
    }

    return options[a_size]


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def ncolors(num):
    rgb_colors = []
    rgb_colors_hex = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
        rgb_colors_hex.append(rgb_to_hex([r, g, b]))

    return rgb_colors, rgb_colors_hex


def rgb_to_hex(rgb):
    if type(rgb) != str:
        for i in range(0, len(rgb)):
            rgb[i] = str(rgb[i])
    else:
        rgb = rgb.split(',')
    heximal = '#'
    for i in rgb:
        num = int(i)
        heximal += str(hex(num))[-2:].replace('x', '0').upper()
    return heximal


def visualization(t, dat_znorm, paa_size=10, alphabet_size=4, sliding_len=20):


    plt.cla()
    plt.figure(figsize=(12, 8))
    plt.plot(dat_znorm)

    paa_x, data_paa, x, y = paa(dat_znorm, paa_size)
    plt.plot(x, y)

    sax_string, sax_plot_x, sax_plot_y, sax_string_plot = sax(data_paa, x, y, breakpoints(alphabet_size))
    for key in sax_plot_y:
        plt.plot(sax_plot_x, sax_plot_y[key])

    # plt.savefig('static/images/' + session['username'] + 'sax.png', bbox_inches='tight')

    sio = BytesIO()
    plt.savefig(sio, format='png', bbox_inches='tight', pad_inches=0.2)
    data = base64.encodebytes(sio.getvalue()).decode()
    chart_mtpl = 'data:image/png;base64,' + str(data)
    plt.close()

    # --------------------------------------
    markline = []
    for i in breakpoints(alphabet_size)[1:]:
        markline.append({"yAxis": i})
    for i in paa_x:
        markline.append({"xAxis": round(i)})

    markpoint = []
    for i in range(0, paa_size):
        markpoint.append(
            MarkPointItem(
                name=sax_string[i],
                symbol='pin',
                itemstyle_opts=ItemStyleOpts(color='#E6E6FA', opacity=0.9),
                coord=[round((2 * i + 1) * len(t) / (2 * paa_size)), data_paa[i]],
                value=i + 1
            )
        )

    chart_sax = (
        Line(init_opts=InitOpts(theme=ThemeType.ROMA, ))

            .add_xaxis(np.linspace(0, len(t) - 1, len(t)))

            .add_yaxis(
            "Z-norm",
            dat_znorm,
            is_symbol_show=False,
        )

            .add_xaxis(x)

            .add_yaxis(
            "PAA",
            y,
            is_symbol_show=False,
            linestyle_opts=LineStyleOpts(
                width=2,
                opacity=0.6,
            ),
            markpoint_opts=MarkPointOpts(
                data=markpoint,
                symbol_size=30,
            )
        )

            .set_global_opts(
            title_opts=TitleOpts(title="SAX Representation"),
            tooltip_opts=TooltipOpts(trigger='axis'),
            datazoom_opts=[
                DataZoomOpts(
                    xaxis_index=0,
                    range_start=0,
                    range_end=100
                ),
                DataZoomOpts(
                    type_='inside',
                    range_start=0,
                    range_end=100,
                ),
                # DataZoomOpts(type_="slider", orient='vertical',is_realtime=False,range_start=0,range_end=100),
                # DataZoomOpts(type_="inside", yaxis_index=0,is_realtime=False)
            ],
            toolbox_opts=ToolboxOpts(pos_left='right',
                                          feature=ToolBoxFeatureOpts(
                                              magic_type=ToolBoxFeatureMagicTypeOpts(is_show=False))),
            brush_opts=BrushOpts()
        )

            .set_series_opts(
            markline_opts=MarkLineOpts(
                data=markline,
                precision=10,
                symbol_size=2,
                linestyle_opts=LineStyleOpts(width=0.6, opacity=0.3, type_='dashed'),
                label_opts=LabelOpts(position='end')
            )
        )
    )

    for key in sax_plot_y:
        chart_sax.add_yaxis(
            key, sax_plot_y[key],
            is_symbol_show=False,
            linestyle_opts=LineStyleOpts(width=6, opacity=0.9),
        )

    # ----------------------------------------
    breakpoints_t = breakpoints(alphabet_size) * np.std(t) + np.mean(t)
    colors = ncolors(alphabet_size)[1]
    split_line = []

    for i in range(0, alphabet_size - 1):
        split_line.append(
            {'gt': breakpoints_t[i],
             'lte': breakpoints_t[i + 1],
             'label': 'β' + str(i) + '-β' + str(i + 1),
             'color': colors[i]
             }
        )
    split_line.append(
        {'gt': breakpoints_t[-1],
         'label': '>β' + str(alphabet_size - 1),
         'color': colors[-1]
         }
    )

    markline_t = []
    for i in breakpoints_t[1:]:
        markline_t.append({"yAxis": i})

    chart_original = (
        Line()

            .add_xaxis(np.linspace(0, len(t) - 1, len(t)))

            .add_yaxis(
            "Original time series",
            t,
            is_symbol_show=False)
            .add_xaxis(x)

            .set_global_opts(
            title_opts=TitleOpts(title="Painted Original Time Series"),
            tooltip_opts=TooltipOpts(trigger='axis', axis_pointer_type="cross"),
            datazoom_opts=[
                DataZoomOpts(xaxis_index=0, range_start=0, range_end=100),
                DataZoomOpts(type_='inside', xaxis_index=0)
            ],
            toolbox_opts=ToolboxOpts(pos_left='right',
                                          feature=ToolBoxFeatureOpts(
                                              magic_type=ToolBoxFeatureMagicTypeOpts(is_show=False))),
            brush_opts=BrushOpts(),
            visualmap_opts=VisualMapOpts(
                pos_top="middle",
                pos_right="0",
                range_opacity=0,
                range_size=[5],
                is_piecewise=True,
                pieces=split_line
            )
        )

            .set_series_opts(
            markline_opts=MarkLineOpts(
                data=markline_t,
                precision=4,
                symbol_size=0,
                linestyle_opts=LineStyleOpts(width=0.5, type_='dashed'),
                label_opts=LabelOpts(position='end')
            )
        )
    )

    for key in sax_plot_y:
        chart_original.add_yaxis(
            key, sax_plot_y[key] * np.std(t) + np.mean(t),
            is_symbol_show=False,
            linestyle_opts=LineStyleOpts(
                width=6,
                opacity=0.5,
                color=colors[ord(key) - 97],
            )
        )

    # ----------------------

    chart_sliding = (
        Line()

            .add_xaxis(np.linspace(0, len(t) - 1, len(t)))

            .add_yaxis(
            "Original time series",
            t,
            is_symbol_show=False)

            .add_xaxis(x)

            .set_global_opts(
            yaxis_opts=AxisOpts(
                max_=round(t.max(), 2),
                min_=round(t.min(), 2),
            ),
            title_opts=TitleOpts(title="Sliding window"),
            tooltip_opts=TooltipOpts(trigger='axis', axis_pointer_type="cross"),
            datazoom_opts=[
                DataZoomOpts(xaxis_index=0,
                                  range_start=None,
                                  range_end=None,
                                  start_value=0,
                                  end_value=sliding_len,
                                  is_zoom_lock=True
                                  ),
                DataZoomOpts(type_='inside',
                                  xaxis_index=0,
                                  is_zoom_lock=True
                                  ),
            ],
            toolbox_opts=ToolboxOpts(pos_left='right',
                                          feature=ToolBoxFeatureOpts(
                                              magic_type=ToolBoxFeatureMagicTypeOpts(is_show=False))),
            brush_opts=BrushOpts(),
            visualmap_opts=VisualMapOpts(
                pos_top="middle",
                pos_right="0",
                range_opacity=0,
                range_size=[5],
                is_piecewise=True,
                pieces=split_line
            )
        )

            .set_series_opts(
            markline_opts=MarkLineOpts(
                data=markline_t,
                precision=4,
                symbol_size=0,
                linestyle_opts=LineStyleOpts(width=0.3, type_='dashed'),
                label_opts=LabelOpts(position='end')
            )
        )
    )

    '''This is the hot-sax finding top 5 discords'''
    num_discords = 5  # initialize the number as 5
    discord = find_discords_hotsax(t, win_size=sliding_len, num_discords=num_discords)
    discord_point = list()
    discord_curve = list()
    
    for i in range(len(discord)):
        idx = discord[i][0]
        dist = discord[i][1]
        empty = np.full(len(t), None, dtype=float)
        empty[idx:idx + sliding_len] = t[idx:idx + sliding_len]
        discord_curve.append(empty)
        discord_point.append(MarkPointItem(name='Discord distance',
                                                symbol='pin',
                                                itemstyle_opts=ItemStyleOpts(color='#CD3333', opacity=0.8),
                                                coord=[idx, t[idx]],
                                                value=round(dist, 1),
                                                )
                             )
    chart_discord = (
        Line(init_opts=InitOpts(theme=ThemeType.ROMA))
            .add_xaxis(np.linspace(0, len(t) - 1, len(t)))
            .add_yaxis(
            "Original time series",
            t,
            is_symbol_show=False,
            is_smooth=True,
            markpoint_opts=MarkPointOpts(
                data=discord_point,
                symbol_size=45,
            ),
            linestyle_opts=LineStyleOpts(
                width=2,
                color='#1C86EE',
                opacity=0.9
            )
        )
            .set_global_opts(
            title_opts=TitleOpts(title="Discords by HOT-SAX"),
            tooltip_opts=TooltipOpts(trigger='axis', axis_pointer_type="cross"),
            datazoom_opts=[
                DataZoomOpts(xaxis_index=0, range_start=0, range_end=100),
                DataZoomOpts(type_='inside', range_start=0, range_end=100)
                # DataZoomOpts(type_="slider", orient='vertical',is_realtime=False,range_start=0,range_end=100),
                # DataZoomOpts(type_="inside", yaxis_index=0,is_realtime=False)
            ],
            toolbox_opts=ToolboxOpts(
                pos_left='right',
                feature=ToolBoxFeatureOpts(
                    magic_type=ToolBoxFeatureMagicTypeOpts(is_show=False))
            ),
            brush_opts=BrushOpts(),
        )
    )

    for i in range(len(discord_curve)):
        chart_discord.add_yaxis(
            'D' + str(i),
            discord_curve[i],
            is_symbol_show=False,
            is_smooth=True,
            linestyle_opts=LineStyleOpts(
                width=6,
                color='#838B8B',
                opacity=0.5
            )
        )

    return chart_sax, chart_original, chart_sliding, sax_string, chart_mtpl, chart_discord
    # return chart_sax, chart_original, chart_sliding, chart_mtpl, sax_string
    # .render()


def sax_via_window(series, win_size, paa_size, alphabet_size=3,
                   nr_strategy='exact', z_threshold=0.01):
    """Simple via window conversion implementation."""
    cuts = breakpoints(alphabet_size)
    sax = defaultdict(list)
    sax_str = defaultdict(str)
    #     vocab = np.full(len(series) - win_size,'',dtype='<U'+str(paa_size+1))
    prev_word = ''

    for i in range(0, len(series) - win_size):

        sub_section = series[i:(i + win_size)]

        paa_rep = paa(sub_section, paa_size)[1]

        curr_word = ts_to_string(paa_rep, cuts)

        if '' != prev_word:
            if 'exact' == nr_strategy and prev_word == curr_word:
                continue
            elif 'mindist' == nr_strategy and \
                    is_mindist_zero(prev_word, curr_word):
                continue

        prev_word = curr_word

        sax[curr_word].append(i)

        if sax_str[curr_word] == "":
            sax_str[curr_word] = str(i)
        else:
            sax_str[curr_word] = sax_str[curr_word] + "," + str(i)

    return sax, sax_str


def ts_to_string(series, breakpoints):
    """
    A straightforward num-to-string conversion.
    @series: np.array; the point of PAA series
    @breakpoints: np.array;  [-np.inf, α-1 breakpoints]
    """
    a_size = len(breakpoints)
    sax = list()
    for i in range(0, len(series)):
        num = series[i]
        # if teh number below 0, start from the bottom, or else from the top
        if(num >= 0):
            j = a_size - 1
            while ((j > 0) and (breakpoints[j] >= num)):
                j = j - 1
            sax.append(convert_to_letter(j))
        else:
            j = 1
            while (j < a_size and breakpoints[j] <= num):
                j = j + 1
            sax.append(convert_to_letter(j-1))
    return ''.join(sax)


def compare_strings(sA, sB, alphabet_size, compress_rate=1):
    """
    Compares two strings based on individual letter distance
    Requires that both strings are the same length
    Suppose compression rate is 1
    Return
    """
    #         if len(sA) != len(sB):
    #             raise StringsAreDifferentLength()
    list_letters_a = [x for x in sA]
    list_letters_b = [x for x in sB]
    mindist = 0.0
    compare_dict = build_letter_compare_dict(alphabet_size)

    for i in range(0, len(list_letters_a)):
        if list_letters_a[i] is not '-' and list_letters_b[i] is not '-':
            mindist += compare_dict[list_letters_a[i] + list_letters_b[i]] ** 2

    mindist = np.sqrt(compress_rate * mindist)

    return mindist


def build_letter_compare_dict(alphabet_size):
    """
    Builds up the lookup table to determine numeric distance between two letters
    given an alphabet size.  Entries for both 'ab' and 'ba' will be created
    and will have identical values.
    """

    compareDict = dict()
    number_rep = range(0, alphabet_size)
    letters = [chr(x + ord('a')) for x in number_rep]

    for i in range(0, len(letters)):
        for j in range(0, len(letters)):
            if np.abs(number_rep[i] - number_rep[j]) <= 1:
                compareDict[letters[i] + letters[j]] = 0
            else:
                high_num = np.max([number_rep[i], number_rep[j]]) - 1
                low_num = np.min([number_rep[i], number_rep[j]])
                compareDict[letters[i] + letters[j]] = breakpoints(alphabet_size)[1:][high_num] - \
                                                       breakpoints(alphabet_size)[1:][low_num]

    return compareDict


def distance_measure(ts1, ts2, paa_size, sax_size):
    data_paa_a, x, y = paa(znorm(ts1), paa_size)[1:]
    a = sax(data_paa_a, x, y, breakpoints(sax_size))[0]

    data_paa_b, x, y = paa(znorm(ts2), paa_size)[1:]
    b = sax(data_paa_b, x, y, breakpoints(sax_size))[0]

    return a, b, compare_strings(a, b, sax_size, len(ts1) / sax_size)


def draw_distance(ts1, ts2, paa_size, sax_size):
    '''
    suppose ts1 and ts2 are already Z-normalized
    '''

    # plot ts1 and ts1 to show euclidean distance
    plt.cla()
    plt.plot(ts1)  # plot ts1
    plt.plot(ts2)  # plot ts2
    sio = BytesIO()
    plt.savefig(sio, format='png')
    data = base64.encodebytes(sio.getvalue()).decode()
    src1 = 'data:image/png;base64,' + str(data)
    plt.close()

    plt.cla()
    # plot ts1 sax segment
    data_paa, x, y = paa(ts1, paa_size)[1:]
    sax_plot_x, sax_plot_y, sax_string_plot= sax(data_paa, x, y, breakpoints(sax_size))[1:]
    for key in sax_plot_y:
        plt.plot(sax_plot_x, sax_plot_y[key])
    # plot ts2 sax segment
    data_paa, x, y = paa(ts2, paa_size)[1:]
    sax_plot_x, sax_plot_y, sax_string_plot = sax(data_paa, x, y, breakpoints(sax_size))[1:]
    for key in sax_plot_y:
        plt.plot(sax_plot_x, sax_plot_y[key])

    sio = BytesIO()
    plt.savefig(sio, format='png')
    data = base64.encodebytes(sio.getvalue()).decode()
    src2 = 'data:image/png;base64,' + str(data)
    plt.close()

    # plot sax_string_plot
    plt.cla()
    for key in sax_plot_y:
        plt.plot(x, sax_string_plot[key])
    plt.yticks(list(range(len(sax_string_plot))), [key for key, value in sax_string_plot.items()])
    sio = BytesIO()
    plt.savefig(sio, format='png')
    data = base64.encodebytes(sio.getvalue()).decode()
    src_string = 'data:image/png;base64,' + str(data)
    plt.close()

    return src1, src2, src_string


def euclidean_distance(ts1, ts2):
    ts1 = znorm(ts1)
    ts2 = znorm(ts2)

    dist = 0

    for i in range(len(ts1)):
        dist += (ts1[i] - ts2[i]) ** 2
    dist = np.sqrt(dist)

    return dist


def paa_distance(ts1, ts2, paa_size):
    ts1 = paa(znorm(ts1), paa_size)[1]
    ts2 = paa(znorm(ts2), paa_size)[1]

    dist = 0
    for i in range(len(ts1)):
        dist += (ts1[i] - ts2[i]) ** 2
    dist = np.sqrt(dist)

    return dist


def find_discords_hotsax(series, win_size=100, num_discords=2, a_size=3,
                         paa_size=3, z_threshold=0.01):
    """HOT-SAX-driven discords discovery."""
    discords = list()

    globalRegistry = set()

    while (len(discords) < num_discords):

        bestDiscord = find_best_discord_hotsax(series, win_size, a_size,
                                               paa_size, z_threshold,
                                               globalRegistry)

        if -1 == bestDiscord[0]:
            break

        discords.append(bestDiscord)

        mark_start = bestDiscord[0] - win_size
        if 0 > mark_start:
            mark_start = 0

        mark_end = bestDiscord[0] + win_size
        '''if len(series) < mark_end:
            mark_end = len(series)'''

        for i in range(mark_start, mark_end):
            globalRegistry.add(i)

    return discords


def find_best_discord_hotsax(series, win_size, a_size, paa_size,
                             znorm_threshold, globalRegistry): # noqa: C901
    """Find the best discord with hotsax."""
    """[1.0] get the sax data first"""
    sax_none = sax_via_window(series, win_size, a_size, paa_size, "none", 0.01)[0]

    """[2.0] build the 'magic' array"""
    magic_array = list()
    for k, v in sax_none.items():
        magic_array.append((k, len(v)))

    """[2.1] sort it desc by the key"""
    m_arr = sorted(magic_array, key=lambda tup: tup[1])

    """[3.0] define the key vars"""
    bestSoFarPosition = -1
    bestSoFarDistance = 0.

    distanceCalls = 0

    visit_array = np.zeros(len(series), dtype=np.int)

    """[4.0] and we are off iterating over the magic array entries"""
    for entry in m_arr:

        """[5.0] some moar of teh vars"""
        curr_word = entry[0]
        occurrences = sax_none[curr_word]

        """[6.0] jumping around by the same word occurrences makes it easier to
        nail down the possibly small distance value -- so we can be efficient
        and all that..."""
        for curr_pos in occurrences:

            if curr_pos in globalRegistry:
                continue

            """[7.0] we don't want an overlapping subsequence"""
            mark_start = curr_pos - win_size
            mark_end = curr_pos + win_size
            visit_set = set(range(mark_start, mark_end))

            """[8.0] here is our subsequence in question"""
            cur_seq = znorm(series[curr_pos:(curr_pos + win_size)],
                            znorm_threshold)

            """[9.0] let's see what is NN distance"""
            nn_dist = np.inf
            do_random_search = 1

            """[10.0] ordered by occurrences search first"""
            for next_pos in occurrences:

                """[11.0] skip bad pos"""
                if next_pos in visit_set:
                    continue
                else:
                    visit_set.add(next_pos)

                """[12.0] distance we compute"""
                dist = euclidean(cur_seq, znorm(series[next_pos:(
                                 next_pos+win_size)], znorm_threshold))
                distanceCalls += 1

                """[13.0] keep the books up-to-date"""
                if dist < nn_dist:
                    nn_dist = dist
                if dist < bestSoFarDistance:
                    do_random_search = 0
                    break

            """[13.0] if not broken above,
            we shall proceed with random search"""
            if do_random_search:
                """[14.0] build that random visit order array"""
                curr_idx = 0
                for i in range(0, (len(series) - win_size)):
                    if not(i in visit_set):
                        visit_array[curr_idx] = i
                        curr_idx += 1
                it_order = np.random.permutation(visit_array[0:curr_idx])
                curr_idx -= 1

                """[15.0] and go random"""
                while curr_idx >= 0:
                    rand_pos = it_order[curr_idx]
                    curr_idx -= 1

                    dist = euclidean(cur_seq, znorm(series[rand_pos:(
                                     rand_pos + win_size)], znorm_threshold))
                    distanceCalls += 1

                    """[16.0] keep the books up-to-date again"""
                    if dist < nn_dist:
                        nn_dist = dist
                    if dist < bestSoFarDistance:
                        nn_dist = dist
                        break

            """[17.0] and BIGGER books"""
            if (nn_dist > bestSoFarDistance) and (nn_dist < np.inf):
                bestSoFarDistance = nn_dist
                bestSoFarPosition = curr_pos

    return bestSoFarPosition, bestSoFarDistance


def euclidean(a, b):
    """Compute a Euclidean distance value."""
    return np.sqrt(np.sum((a-b)**2))