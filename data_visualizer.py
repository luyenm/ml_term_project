import matplotlib.pyplot as plt


# Generates a bar chart.
# x_axis - the X axis of the chart
# y_axis - the Y axis of the chart
# xlabel - the label of the x axis
# ylabel - the label of the y axis
def generate_bar_chart(x_axis, y_axis, xlabel, ylabel):
    title = "Bar chart for: " + xlabel + " vs. " + ylabel
    plt.bar(x_axis, y_axis)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


# Generates a scatterplot
# x_axis - the X axis of the chart
# y_axis - the Y axis of the chart
# xlabel - the label of the x axis
# ylabel - the label of the y axis
def generate_scatterplot(x_axis, y_axis, xlabel, ylabel):
    title = "Scatterplot for: " + xlabel + " vs. " + ylabel
    plt.scatter(x_axis, y_axis)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# Generates a scatterplot
# x_axis - the X axis of the chart
# xlabel - the label of the x axis
def generate_histogram(x_axis, xlabel):
    title = "Histogram for: " + xlabel
    plt.hist(x_axis)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()


# Save scatterplot as a png image, to a folder called 'scatterplot'
# x_axis - the X axis of the chart
# y_axis - the Y axis of the chart
# xlabel - the label of the x axis
# ylabel - the label of the y axis
def image_scatterplot(x_axis, y_axis, xlabel, ylabel):
    title = "Bar chart for: " + xlabel + " vs. " + ylabel
    plt.scatter(x_axis, y_axis)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(fname="scatterplot" + '\\' + xlabel + '_vs_' + ylabel + '.png', format='png')
    plt.close('all')


# Save histogram as a png image, to a folder called 'histogram'
# x_axis - the X axis of the chart
# xlabel - the label of the x axis
# prefix - prefix to filename
def image_histogram(x_axis, xlabel, prefix):
    title = "Histogram for: " + xlabel
    plt.hist(x_axis)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(fname="histogram" + '\\' + prefix + xlabel + '.png', format='png')
    plt.close('all')


# This is used for K-Fold validation
# x-axis
# y-axis
def plot_line_graph(x_axis, x_axis2, y_axis, xlabel, ylabel, title):
    plt.plot(y_axis, x_axis)
    plt.plot(y_axis, x_axis2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(fname="line plot" + '\\' + xlabel + '_vs_' + ylabel + '.png', format='png')
    plt.close('all')
