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
# y_axis - the Y axis of the chart
# xlabel - the label of the x axis
# ylabel - the label of the y axis
def generate_histogram(x_axis, y_axis, xlabel, ylabel):
    title = "Scatterplot for: " + xlabel + " vs. " + ylabel
    plt.hist(x_axis, y_axis)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()