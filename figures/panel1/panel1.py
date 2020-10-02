import matplotlib.pyplot as plt
import numpy as np
import pylustrator

pylustrator.start()

pylustrator.load("setup_notext.png", dpi=600)
pylustrator.load("channel.png", dpi=600)
pylustrator.load("ellipse_test.py")
pylustrator.load("cells.png")
pylustrator.load("analytical_solution.py")


plt.axes([0.2, 0.2, 0.3, 0.4], facecolor="none", label="overlay")

plt.gca().set_axis_off()
import matplotlib.patches as mpatches
circle = mpatches.Circle([2, 2], 1, ec="none", facecolor="C1", alpha=0.5)
plt.gca().add_patch(circle)
circle = mpatches.Circle([2, 2], 1, ec="black", facecolor="none", lw=0.8)
plt.gca().add_patch(circle)
plt.plot([2, 2], [0, 2], "k", lw=0.5)
plt.plot([3, 3], [0, 2], "k", lw=0.5)

plt.plot([0, 2], [2, 2], "k", lw=0.5)
plt.plot([0, 2], [3, 3], "k", lw=0.5)
plt.xlim(0, 3.4)
plt.ylim(0, 3.4)
#plt.axis("equal")

hw = 0.3
hl = 0.3
xoffset = 0.85
len = 0.55
plt.arrow(2-xoffset, 0.3, len, -0, length_includes_head=True, head_starts_at_zero=True, head_width=hw, head_length=hl, fc="k")
plt.arrow(3+xoffset, 0.3,-len, -0, length_includes_head=True, head_starts_at_zero=True, head_width=hw, head_length=hl, fc="k")

plt.arrow(0.3, 2-xoffset, 0, len, length_includes_head=True, head_starts_at_zero=True, head_width=hw, head_length=hl, fc="k")
plt.arrow(0.32, 3+xoffset, 0, -len, length_includes_head=True, head_starts_at_zero=True, head_width=hw, head_length=hl, fc="k")


#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(16.980000/2.54, 11.000000/2.54, forward=True)
plt.figure(1).ax_dict["cells.png"].set_position([0.765161, 0.025269, 0.216519, 0.539848])
plt.figure(1).ax_dict["cells.png"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["cells.png"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["cells.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[4].transAxes)  # id=plt.figure(1).axes[4].texts[0].new
plt.figure(1).ax_dict["cells.png"].texts[0].set_position([-0.172482, 0.995152])
plt.figure(1).ax_dict["cells.png"].texts[0].set_text("f")
plt.figure(1).ax_dict["cells.png"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["cells.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["cells.png"].transAxes)  # id=plt.figure(1).ax_dict["cells.png"].texts[1].new
plt.figure(1).ax_dict["cells.png"].texts[1].set_ha("center")
plt.figure(1).ax_dict["cells.png"].texts[1].set_position([0.154651, 1.032698])
plt.figure(1).ax_dict["cells.png"].texts[1].set_text("1 bar")
plt.figure(1).ax_dict["cells.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["cells.png"].transAxes)  # id=plt.figure(1).ax_dict["cells.png"].texts[2].new
plt.figure(1).ax_dict["cells.png"].texts[2].set_ha("center")
plt.figure(1).ax_dict["cells.png"].texts[2].set_position([0.505313, 1.032698])
plt.figure(1).ax_dict["cells.png"].texts[2].set_text("2 bar")
plt.figure(1).ax_dict["cells.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["cells.png"].transAxes)  # id=plt.figure(1).ax_dict["cells.png"].texts[3].new
plt.figure(1).ax_dict["cells.png"].texts[3].set_ha("center")
plt.figure(1).ax_dict["cells.png"].texts[3].set_position([0.855975, 1.032698])
plt.figure(1).ax_dict["cells.png"].texts[3].set_text("3 bar")
plt.figure(1).ax_dict["cells.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["cells.png"].transAxes)  # id=plt.figure(1).ax_dict["cells.png"].texts[4].new
plt.figure(1).ax_dict["cells.png"].texts[4].set_ha("right")
plt.figure(1).ax_dict["cells.png"].texts[4].set_position([-0.025992, 0.503692])
plt.figure(1).ax_dict["cells.png"].texts[4].set_text("0")
plt.figure(1).ax_dict["cells.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["cells.png"].transAxes)  # id=plt.figure(1).ax_dict["cells.png"].texts[5].new
plt.figure(1).ax_dict["cells.png"].texts[5].set_ha("right")
plt.figure(1).ax_dict["cells.png"].texts[5].set_position([-0.025992, 0.635424])
plt.figure(1).ax_dict["cells.png"].texts[5].set_text("25")
plt.figure(1).ax_dict["cells.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["cells.png"].transAxes)  # id=plt.figure(1).ax_dict["cells.png"].texts[6].new
plt.figure(1).ax_dict["cells.png"].texts[6].set_ha("right")
plt.figure(1).ax_dict["cells.png"].texts[6].set_position([-0.025992, 0.773379])
plt.figure(1).ax_dict["cells.png"].texts[6].set_text("50")
plt.figure(1).ax_dict["cells.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["cells.png"].transAxes)  # id=plt.figure(1).ax_dict["cells.png"].texts[7].new
plt.figure(1).ax_dict["cells.png"].texts[7].set_ha("right")
plt.figure(1).ax_dict["cells.png"].texts[7].set_position([-0.025992, 0.912150])
plt.figure(1).ax_dict["cells.png"].texts[7].set_text("75")
plt.figure(1).ax_dict["cells.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["cells.png"].transAxes)  # id=plt.figure(1).ax_dict["cells.png"].texts[8].new
plt.figure(1).ax_dict["cells.png"].texts[8].set_ha("right")
plt.figure(1).ax_dict["cells.png"].texts[8].set_position([-0.025992, 0.378183])
plt.figure(1).ax_dict["cells.png"].texts[8].set_text("25")
plt.figure(1).ax_dict["cells.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["cells.png"].transAxes)  # id=plt.figure(1).ax_dict["cells.png"].texts[9].new
plt.figure(1).ax_dict["cells.png"].texts[9].set_ha("right")
plt.figure(1).ax_dict["cells.png"].texts[9].set_position([-0.025992, 0.236080])
plt.figure(1).ax_dict["cells.png"].texts[9].set_text("50")
plt.figure(1).ax_dict["cells.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["cells.png"].transAxes)  # id=plt.figure(1).ax_dict["cells.png"].texts[10].new
plt.figure(1).ax_dict["cells.png"].texts[10].set_ha("right")
plt.figure(1).ax_dict["cells.png"].texts[10].set_position([-0.025992, 0.085679])
plt.figure(1).ax_dict["cells.png"].texts[10].set_text("75")
plt.figure(1).ax_dict["cells.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["cells.png"].transAxes)  # id=plt.figure(1).ax_dict["cells.png"].texts[11].new
plt.figure(1).ax_dict["cells.png"].texts[11].set_ha("center")
plt.figure(1).ax_dict["cells.png"].texts[11].set_position([-0.254454, 0.266533])
plt.figure(1).ax_dict["cells.png"].texts[11].set_rotation(90.0)
plt.figure(1).ax_dict["cells.png"].texts[11].set_text("radial position (µm)")
plt.figure(1).ax_dict["cells.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["cells.png"].transAxes)  # id=plt.figure(1).ax_dict["cells.png"].texts[12].new
plt.figure(1).ax_dict["cells.png"].texts[12].set_fontsize(8)
plt.figure(1).ax_dict["cells.png"].texts[12].set_position([0.822459, -0.008210])
plt.figure(1).ax_dict["cells.png"].texts[12].set_text("10 µm")
plt.figure(1).ax_dict["cells.png"].get_xaxis().get_label().set_text("")
plt.figure(1).ax_dict["cells.png"].get_yaxis().get_label().set_text("")
plt.figure(1).ax_dict["channel.png"].set_position([0.780942, 0.658657, 0.200738, 0.332516])
plt.figure(1).ax_dict["channel.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["channel.png"].transAxes)  # id=plt.figure(1).ax_dict["channel.png"].texts[0].new
plt.figure(1).ax_dict["channel.png"].texts[0].set_position([-0.008846, 0.846099])
plt.figure(1).ax_dict["channel.png"].texts[0].set_text("b")
plt.figure(1).ax_dict["channel.png"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["overlay"].set_position([0.337827, 0.433040, 0.066033, 0.101873])
plt.figure(1).ax_dict["overlay"].set_xlim(0.0, 3.8)
plt.figure(1).ax_dict["overlay"].set_ylim(0.0, 3.8)
plt.figure(1).ax_dict["setup_notext.png"].set_position([0.026541, 0.543344, 0.727489, 0.447876])
plt.figure(1).ax_dict["setup_notext.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["setup_notext.png"].transAxes)  # id=plt.figure(1).ax_dict["setup_notext.png"].texts[0].new
plt.figure(1).ax_dict["setup_notext.png"].texts[0].set_ha("left")
plt.figure(1).ax_dict["setup_notext.png"].texts[0].set_position([-0.033430, 0.395054])
plt.figure(1).ax_dict["setup_notext.png"].texts[0].set_rotation(0.0)
plt.figure(1).ax_dict["setup_notext.png"].texts[0].set_text("compr.\nair")
plt.figure(1).ax_dict["setup_notext.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["setup_notext.png"].transAxes)  # id=plt.figure(1).ax_dict["setup_notext.png"].texts[1].new
plt.figure(1).ax_dict["setup_notext.png"].texts[1].set_position([0.149486, 0.343334])
plt.figure(1).ax_dict["setup_notext.png"].texts[1].set_text("pressure\nregulator")
plt.figure(1).ax_dict["setup_notext.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["setup_notext.png"].transAxes)  # id=plt.figure(1).ax_dict["setup_notext.png"].texts[2].new
plt.figure(1).ax_dict["setup_notext.png"].texts[2].set_position([-0.016224, 0.939596])
plt.figure(1).ax_dict["setup_notext.png"].texts[2].set_text("a")
plt.figure(1).ax_dict["setup_notext.png"].texts[2].set_weight("bold")
plt.figure(1).ax_dict["setup_notext.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["setup_notext.png"].transAxes)  # id=plt.figure(1).ax_dict["setup_notext.png"].texts[3].new
plt.figure(1).ax_dict["setup_notext.png"].texts[3].set_position([0.270994, 0.809433])
plt.figure(1).ax_dict["setup_notext.png"].texts[3].set_text("3-way\nvalve")
plt.figure(1).ax_dict["setup_notext.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["setup_notext.png"].transAxes)  # id=plt.figure(1).ax_dict["setup_notext.png"].texts[4].new
plt.figure(1).ax_dict["setup_notext.png"].texts[4].set_ha("center")
plt.figure(1).ax_dict["setup_notext.png"].texts[4].set_position([0.405703, 0.100550])
plt.figure(1).ax_dict["setup_notext.png"].texts[4].set_text("cell\nreservoir")
plt.figure(1).ax_dict["setup_notext.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["setup_notext.png"].transAxes)  # id=plt.figure(1).ax_dict["setup_notext.png"].texts[5].new
plt.figure(1).ax_dict["setup_notext.png"].texts[5].set_ha("center")
plt.figure(1).ax_dict["setup_notext.png"].texts[5].set_position([0.659406, 0.657529])
plt.figure(1).ax_dict["setup_notext.png"].texts[5].set_text("slide with $200\\times200$µm\nmicrochannels")
plt.figure(1).ax_dict["setup_notext.png"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["setup_notext.png"].transAxes)  # id=plt.figure(1).ax_dict["setup_notext.png"].texts[6].new
plt.figure(1).ax_dict["setup_notext.png"].texts[6].set_ha("center")
plt.figure(1).ax_dict["setup_notext.png"].texts[6].set_position([0.877187, 0.274958])
plt.figure(1).ax_dict["setup_notext.png"].texts[6].set_text("waste")
plt.figure(1).axes[2].set_position([0.388137, 0.269430, 0.341050, 0.249317])
plt.figure(1).axes[2].texts[1].set_position([0.129738, 0.500001])
plt.figure(1).axes[2].texts[2].set_fontsize(8)
plt.figure(1).axes[2].texts[3].set_fontsize(8)
plt.figure(1).axes[2].texts[4].set_fontsize(8)
plt.figure(1).axes[2].texts[4].set_position([-0.150000, -0.512969])
plt.figure(1).axes[2].texts[5].set_fontsize(8)
plt.figure(1).axes[2].texts[5].set_position([1.188659, 0.055460])
plt.figure(1).axes[2].texts[6].set_fontsize(8)
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[7].new
plt.figure(1).axes[2].texts[7].set_position([0.201521, 0.778511])
plt.figure(1).axes[2].texts[7].set_text("d")
plt.figure(1).axes[2].texts[7].set_weight("bold")
plt.figure(1).axes[3].set_position([0.469724, 0.041503, 0.183585, 0.277359])
plt.figure(1).axes[3].texts[0].set_fontsize(8)
plt.figure(1).axes[3].texts[0].set_position([4.685398, 0.196495])
plt.figure(1).axes[3].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[3].transAxes)  # id=plt.figure(1).axes[3].texts[1].new
plt.figure(1).axes[3].texts[1].set_position([-0.070040, 0.631857])
plt.figure(1).axes[3].texts[1].set_text("e")
plt.figure(1).axes[3].texts[1].set_weight("bold")
plt.figure(1).axes[5].set_position([0.098897, 0.143082, 0.292987, 0.378555])
plt.figure(1).axes[5].spines['right'].set_visible(False)
plt.figure(1).axes[5].spines['top'].set_visible(False)
plt.figure(1).axes[5].get_legend()._set_loc((0.040413, 0.427214))
plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[0].new
plt.figure(1).axes[5].texts[0].set_position([-0.218291, 1.011572])
plt.figure(1).axes[5].texts[0].set_text("c")
plt.figure(1).axes[5].texts[0].set_weight("bold")
plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[1].new
plt.figure(1).axes[5].texts[1].set_fontsize(7)
plt.figure(1).axes[5].texts[1].set_position([0.598559, 0.932262])
plt.figure(1).axes[5].texts[1].set_text("$\sigma$ ~ 5%")
plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[2].new
plt.figure(1).axes[5].texts[2].set_fontsize(8)
plt.figure(1).axes[5].texts[2].set_position([0.883209, 0.676723])
plt.figure(1).axes[5].texts[2].set_text("$r$ ~ 8 µm")
plt.figure(1).axes[5].get_xaxis().get_label().set_text("distance from channel center ($\mu m$)")
plt.figure(1).axes[5].get_yaxis().get_label().set_text("shear stress (Pa)")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3] + ".png", dpi=300)
plt.savefig(__file__[:-3] + ".pdf")
plt.show()