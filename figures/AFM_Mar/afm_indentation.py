import numpy as np
import matplotlib.pyplot as plt
import pylustrator
pylustrator.start()

#plt.plot([1, 2], [2, 3])
#plt.show()
import jpkfile

"""
file = jpkfile.JPKFile("force-save-2021.07.05-15.47.05.396.jpk-force")

print(file.segments[0].data.keys())
plt.plot(-file.segments[0].get_decoded_data('height')[0] * 1e9 + 3000,
         file.segments[0].get_decoded_data('vDeflection')[0] * 1e12)
plt.plot(-file.segments[1].get_decoded_data('height')[0] * 1e9 + 3000,
         file.segments[1].get_decoded_data('vDeflection')[0] * 1e12)
# plt.plot(file.segments[1].get_decoded_data('height')[0], file.segments[1].get_decoded_data('vDeflection')[0])
plt.show()
"""

def plotFig(filename, fignr=1):
    from scipy.io import loadmat
    from numpy import size
    d = loadmat(filename, squeeze_me=True, struct_as_record=False)
    ax1 = d['hgS_070000'].children
    if size(ax1) > 1:
        legs = ax1[1]
        ax1 = ax1[0]
    else:
        legs = 0
    #plt.figure(fignr)
    #plt.clf()
    # hold(True)
    counter = 0
    for line in ax1.children:
        if line.type == 'graph2d.lineseries':
            if hasattr(line.properties, 'Marker'):
                mark = "%s" % line.properties.Marker
                mark = mark[0]
            else:
                mark = '.'
            if hasattr(line.properties, 'LineStyle'):
                linestyle = "%s" % line.properties.LineStyle
            else:
                linestyle = '-'
            if hasattr(line.properties, 'Color'):
                r, g, b = line.properties.Color
            else:
                r = 0
                g = 0
                b = 1
            if hasattr(line.properties, 'MarkerSize'):
                marker_size = line.properties.MarkerSize
            else:
                marker_size = 1
            x = line.properties.XData
            y = line.properties.YData
            plt.plot(x, y)#, marker=mark, linestyle=linestyle, markersize=marker_size)#color=(r, g, b),
        elif line.type == 'text':
            if counter < 1:
                plt.xlabel("%s" % line.properties.String)#, fontsize=16)
                counter += 1
            elif counter < 2:
                plt.ylabel("%s" % line.properties.String)#, fontsize=16)
                counter += 1
    plt.xlim(ax1.properties.XLim)
    plt.ylim(ax1.properties.YLim)



plotFig("Fzvisco2.5e+04Curves4DCpaper2111091720.fig")
plt.show()
