import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import QuadMesh
from matplotlib.image import AxesImage

#Also use this region to write some default parameters for the plotting
plt.rcParams['figure.figsize'] = [6, 2]


def plot_norm(x, y, *args, mode="max", **kwargs):
    if mode == 'max': 
        plt.plot(x, y/max(y), *args, **kwargs)
    if mode == 'sum': 
        plt.plot(x, y/sum(y), *args, **kwargs)
        
def plot_range(x, y, *args, **kwargs):
    """
    In kwargs, there are "hidden" parameters that can be used to manipulate the plot_norm function!
    max: This sets the max value of plot
    shift: This will shift the plot
    """
    #write a if statement if kwargs has the key "max"

    if "max" in kwargs.keys():
        max = kwargs["max"]
        kwargs.pop("max") #remove the key "max" from kwargs
    else:
        max = 1

    if "min" in kwargs.keys():
        min = kwargs["min"]
        kwargs.pop("min") #remove the key "max" from kwargs
    else:
        min = 0

    ymin = np.min(y)
    ymax = np.max(y)
    y_proc = min + (y - ymin) * (max - min) / (ymax - ymin)

    plt.plot(x, y_proc, *args, **kwargs)
    
def plot_norm_y_only(y, *args, **kwargs):
    plt.plot(y/np.max(y), *args, **kwargs)
    
def custom_figure(*args, **kwargs):
    fig = plt.figure(*args, **kwargs)
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    
    plt.tight_layout()
    return fig

def meshgrid_center(xdata, ydata):
    x = (xdata[:-1]+xdata[1:])/2
    x = np.array([2*xdata[0]-x[0], *x, 2*xdata[-1]-x[-1]])
    y = (ydata[:-1]+ydata[1:])/2
    y = np.array([2*ydata[0]-y[0], *y, 2*ydata[-1]-y[-1]])
    return x, y

def pcolormesh_center(x, y, Z, **kws):
    X, Y = meshgrid_center(x, y)
    return plt.pcolormesh(X, Y, Z.T, **kws)

## Plotting functions


def plot_norm_col(xs_list, ys_list, args_list=None, kwargs_list=None, title_list=None, col_width=5, row_height=1, suptitle=None, norm_flag=True):
    """
    Example use case:
    xs_list = [xaxis_cam, x_3F, x_3F]
    ys_list = [output_list, output_th_list, input_th_list]
    args_list = ["", "--", "--k"]
    kwargs_list = [dict(c="tab:blue", label="output"), 
                   dict(c="tab:red", alpha=0.7, label="output theory"), 
                   dict(alpha=0.3, label="input theory")]
    title_list = [f"shift={shift:.2f}" for shift in shift_list]

    fig, axs = plot_norm_col(xs_list, ys_list, args_list, kwargs_list, title_list);
    [ax.set_xlim(-0.5, 0.5) for ax in axs];
    """
    Nys = len(ys_list[0])
    Nlines = len(xs_list)
    if args_list is None: #note that args_list is a bit of a bad name. This is because plots only take in 1 arg, so I can trivially do the job of making the construction for you in this code.
        args_list = [[] for i in range(Nlines)]
    else:
        args_list = [[arg] for arg in args_list]
    if kwargs_list is None: kwargs_list = [{} for i in range(Nlines)]

    figsize = (col_width, Nys*row_height)
    fig, axs = plt.subplots(Nys, 1, figsize=figsize, sharex=True, sharey=True, dpi=100)
    fig.subplots_adjust(hspace=0.0)
    
    for (ax_ind, ax) in enumerate(axs):
        plt.sca(ax)
        for l_ind in range(Nlines): #l stands for line here!
            if norm_flag:
                plot_norm(xs_list[l_ind], ys_list[l_ind][ax_ind], *args_list[l_ind], **kwargs_list[l_ind])
            else:
                plt.plot(xs_list[l_ind], ys_list[l_ind][ax_ind], *args_list[l_ind], **kwargs_list[l_ind])
        
        plt.ylim(-0.1, 1.1)
        plt.grid(alpha=0.3)
            
        if title_list is not None:
            plt.text(1.03, 0.5, title_list[ax_ind], horizontalalignment='left',
                     verticalalignment='center', transform=ax.transAxes);
       
    legend_flag = any([ax.lines[0].get_label() != f"_line{i}" for i in range(Nlines)]) #only call function if labels exist
    if legend_flag: 
        axs[-1].legend(fontsize=9, ncol=2, bbox_to_anchor=[0.5, -1.2/row_height], loc='lower center')

    if suptitle is not None:
        axs[0].set_title(suptitle, fontsize=11)

    return fig, axs

def plot_norm_2col(xs1_list, xs2_list, ys1_list, ys2_list, args_list=None, kwargs_list=None, title_list=None, col_width=5, row_height=1, suptitles=None, norm_flag=True):
    #throw an error is len of ys1_list and ys2_list are not the same
    if len(ys1_list) != len(ys2_list):
        raise ValueError("len(ys1_list) != len(ys2_list)")

    Nys = max(len(ys1_list[0]), len(ys2_list[0]))
    Nlines = max(len(xs1_list), len(xs2_list))

    if args_list is None: 
        args_list = [[] for i in range(Nlines)]
    else:
        args_list = [[arg] for arg in args_list]
    if kwargs_list is None: kwargs_list = [{} for i in range(Nlines)]

    figsize = (2*col_width, Nys*row_height)
    fig, axs = plt.subplots(Nys, 2, figsize=figsize, sharex="col", sharey=True, dpi=100)
    fig.subplots_adjust(hspace=0.0)
    
    xs_list_list = [xs1_list, xs2_list]
    ys_list_list = [ys1_list, ys2_list]

    for col_ind in range(2):
        for (ax_ind, ax) in enumerate(axs[:, col_ind]):
            plt.sca(ax)
            for l_ind in range(Nlines): #l stands for line here!
                if norm_flag:
                    plot_norm(xs_list_list[col_ind][l_ind], ys_list_list[col_ind][l_ind][ax_ind], *args_list[l_ind], **kwargs_list[l_ind])
                else:
                    plt.plot(xs_list_list[col_ind][l_ind], ys_list_list[col_ind][l_ind][ax_ind], *args_list[l_ind], **kwargs_list[l_ind])
            
            plt.ylim(-0.1, 1.1)
            plt.grid(alpha=0.3)

            if col_ind == 1:        
                if title_list is not None:
                    plt.text(1.03, 0.5, title_list[ax_ind], horizontalalignment='left',
                            verticalalignment='center', transform=ax.transAxes);
       
    legend_flag = any([axs[0,0].lines[0].get_label() != f"_line{i}" for i in range(Nlines)]) #only call function if labels exist
    if legend_flag: 
        axs[-1,0].legend(fontsize=9, ncol=10, bbox_to_anchor=[1.0, -1.2/row_height], loc='lower center')

    if suptitles is not None:
        axs[0, 0].set_title(suptitles[0], fontsize=11)
        axs[0, 1].set_title(suptitles[1], fontsize=11)

    return fig, axs



def plot_norm_matrix(xs_list, ys_mat_list, args_list=None, kwargs_list=None, xlabel_list=None, ylabel_list=None, col_width=5, row_height=1):
    """
    ys_mat_list is a list of matrices whose values are again vectors! (I know it is confusing...)
    
    Some demo code
    ys_mat_list = [output_cuts, output_conts, input_ths, output_cont_ths]
    xs_list = [xaxis_cam]*2 + [x_3F]*2

    args_list = ["", "", "k--", "m--"]
    kwargs_list = [dict(c="tab:blue", label="cut on", alpha=0.6), 
                   dict(c="tab:red", label="control", alpha=0.6),
                   dict(alpha=0.1, label="input theory"),
                   dict(alpha=0.3, label="output control theory")]

    xlabel_list = [f"shift={shift:g}" for shift in shift_list]
    ylabel_list = [f"angle={angle:g}" for angle in angle_list]

    fig, axs = plot_norm_matrix(xs_list, ys_mat_list, args_list, kwargs_list, xlabel_list, ylabel_list)

    [ax.set_xlim(-0.7, 0.7) for ax in np.ravel(axs)];

    fig.suptitle(f"top, 420, w0=40um", y=0.93)
    """

    Nrow, Ncol, _ = ys_mat_list[0].shape
    Nlines = len(xs_list)

    if args_list is None: #note that args_list is a bit of a bad name. This is because plots only take in 1 arg, so I can trivially do the job of making the construction for you in this code.
        args_list = [[] for i in range(Nlines)]
    else:
        args_list = [[arg] for arg in args_list]
        
    if kwargs_list is None: kwargs_list = [{} for i in range(Nlines)]

    figsize=(col_width*Ncol, row_height*Nrow)
    fig, axs = plt.subplots(Nrow, Ncol, figsize=figsize, sharex=True, sharey=True, dpi=100)
    fig.subplots_adjust(wspace=0, hspace=0)

    for row_ind in range(Nrow):
        for col_ind in range(Ncol):
            plt.sca(axs[row_ind, col_ind])
            for l_ind in range(Nlines): #l stands for line here!
                plot_norm(xs_list[l_ind], ys_mat_list[l_ind][row_ind, col_ind], *args_list[l_ind], **kwargs_list[l_ind])

            plt.ylim(-0.1, 1.1)
            plt.grid(alpha=0.3)

    ### Now run following code for each for the subplots, having making the plots themself
    legend_flag = any([axs[0,0].lines[i].get_label() != f"_line{i}" for i in range(Nlines)]) #only call function if labels exist
    if legend_flag:
        legend_col_ind = Ncol//2
        axs[-1, legend_col_ind].legend(fontsize=10, ncol=Nlines, bbox_to_anchor=[0.5, -1.3], loc='lower center')

    if xlabel_list is not None:
        for (ind, xlabel) in enumerate(xlabel_list):
            ax = axs[0, ind]
            plt.text(0.5, 1.1, xlabel, horizontalalignment='center',
                     verticalalignment='bottom', transform=ax.transAxes)

    if ylabel_list is not None:
        for (ind, ylabel) in enumerate(ylabel_list):
            ax = axs[ind, 0]
            plt.text(-0.15, 0.5, ylabel, horizontalalignment='right',
                     verticalalignment='center', transform=ax.transAxes)
            
    return fig, axs

def plot_grid(x, y_pred, y, ylim=None, xlim=None):
    fig, axs = plt.subplots(3, 3, figsize=(11, 5))
    for (ind, ax) in enumerate(np.ndarray.flatten(axs)):
        ind = ind 
        plt.sca(ax)
        plt.plot(x, y_pred[ind], color="tab:blue", alpha=0.8, label="pred")
        plt.plot(x, y[ind], color="tab:red", alpha=0.7, label="test")
        plt.grid()
        if ylim is not None:
            plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(-xlim, xlim)
        
    plt.legend()
    return fig, axs
    
#### 

def autoscale_y(ax,margin=0.1):
    """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

    import numpy as np

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo,hi = ax.get_xlim()
        try:
            lo = lo*xd.unit
            hi = hi*xd.unit
        except: 
            pass
        y_displayed = yd[((xd>lo) & (xd<hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed)-margin*h
        top = np.max(y_displayed)+margin*h
        return bot,top

    lines = ax.get_lines()
    bot,top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot,top)

def autoscale_vmin_vmax(ax=None, xaxis=None, yaxis=None):
    """
    rescales the vmin and vmax of a pcolormesh or imshow plot based on the data that is visible
    given the current xlim and ylim of the axis.
    ax -- a matplotlib axes object
    """
    if ax is None:
        ax = plt.gca()

    # get the current xlim and ylim
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # find the pcolormesh or imshow object
    img = None
    for child in ax.get_children():
        if isinstance(child, (QuadMesh, AxesImage)):
            img = child
            break

    if img is None:
        print("no pcolormesh or imshow object found in the given axis.")

    if isinstance(img, AxesImage): # imshow object
        # get data that is plotted
        displayed_data = img.get_array()

        if xaxis is None or yaxis is None:
            # get xaxis and yaxis
            extent = img.get_extent()
            #make sure that xaxis is always increasing
            if extent[0] < extent[1]:
                xaxis = np.linspace(extent[0], extent[1], displayed_data.shape[1])
            else:
                xaxis = np.linspace(extent[1], extent[0], displayed_data.shape[1])

            if extent[2] < extent[3]:
                yaxis = np.linspace(extent[2], extent[3], displayed_data.shape[0])
            else:
                yaxis = np.linspace(extent[3], extent[2], displayed_data.shape[0])

    else:  # quadmesh : aka pcolormesh object here
        # get data that is plotted
        displayed_data = img.get_array().data.reshape(img._meshHeight, img._meshWidth)

        if xaxis is None or yaxis is None:
            xaxis = np.zeros(img._meshWidth)
            yaxis = np.zeros(img._meshHeight)
            paths = img.get_paths()

            for i in range(len(xaxis)):
                ind_1d = np.ravel_multi_index((0, i), (img._meshHeight, img._meshWidth))
                xaxis[i] = paths[ind_1d].vertices[:, 0].mean()

            for i in range(len(yaxis)):
                ind_1d = np.ravel_multi_index((i, 0), (img._meshHeight, img._meshWidth))
                yaxis[i] = paths[ind_1d].vertices[:, 1].mean()

    # get the indices corresponding to the current xlim and ylim
    xind_left = np.argmin(np.abs(xaxis - xlim[0]))
    xind_right = np.argmin(np.abs(xaxis - xlim[1]))
    yind_bottom = np.argmin(np.abs(yaxis - ylim[0]))
    yind_top = np.argmin(np.abs(yaxis - ylim[1]))

    # get the visible data based on the indices
    visible_data = displayed_data[yind_bottom:yind_top, xind_left:xind_right]

    # calculate vmin and vmax
    vmin = visible_data.min()
    vmax = visible_data.max()

    # set the new vmin and vmax
    img.set_clim(vmin, vmax)