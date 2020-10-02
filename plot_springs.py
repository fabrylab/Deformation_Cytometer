import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import scipy.optimize

def MakeFromPolar(r: float, theta: float, phi: float) -> np.ndarray:
    """
    Convert from polar coordinates to cartesian coordinates
    """
    # get sine and cosine of the angles
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    # and convert to cartesian coordinates
    x = r * st * cp
    y = r * st * sp
    z = r * ct
    # create the array
    return np.array([x, y, z])

def buildBeams(N: int) -> np.ndarray:
    """
    Builds a sphere of unit vectors with N beams in the xy plane.
    """
    N = int(np.floor(np.sqrt(int(N) * np.pi + 0.5)))

    # start with an empty list
    beams = []

    connections = []

    old_indices = []
    # iterate over the whole angle in the xy plane
    for i in range(N):
        # get the Nth part of the total rotation
        theta = (2 * np.pi / N) * i

        # estimate how many vectors we need to cover the phi angle (for the z direction)
        jmax = int(np.floor(N * np.sin(theta) + 0.5))

        offset = len(beams)
        # iterate over those angles to get beams in every direction
        for j in range(jmax):
            # get the phi angle
            phi = (2 * np.pi / jmax) * j

            # and create a unit vector from the polar coordinates theta and phi
            beams.append([theta, phi])
            if j == 0:
                connections.append([offset + j, offset + j + jmax - 1])
            else:
                connections.append([offset+j-1, offset+j])
        indices = np.arange(offset, offset+jmax)
        a, b = old_indices, indices
        if len(indices) < len(old_indices):
            b, a = old_indices, indices
        if len(a):
            start = 0
            for index, j in enumerate(b):
                start = int(np.floor((index)*len(a)/len(b)))
                end = int(np.floor((index+1)*len(a)/len(b)))
                print(len(a), len(b), len(a)/len(b))
                print("->", start, end)
                for i in (list(a)+list(a))[start:end+1]:
                    connections.append([i, j])
                    print("add", connections[-1])
                start = end# - 1

            #break

        old_indices = indices

    # return all the vectors
    return np.array(beams), connections

def getEllipsoidPoints(N):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    beams, connections = buildBeams(N)
    print(beams.shape)
    points = MakeFromPolar(1, beams[:, 0], beams[:, 1]).T
    print(points.shape)
    #for p in points:
    #    plt.plot([0, p[0]], [0, p[1]], [0, p[2]], "r-")
    for c in connections:
        p0 = points[c[0]]
        p1 = points[c[1]]
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], "r-")
    plt.plot(points[:, 0], points[:, 1], points[:, 2], "o")
    #plt.axis("equal")
    plt.show()

getEllipsoidPoints(35)
exit()

def angles_in_ellipse(
        num,
        a,
        b):
    assert(num > 0)
    if a == b:
        return np.arange(num)/num*np.pi*2
    assert(a < b)
    angles = 2 * np.pi * np.arange(num) / num
    if a != b:
        e = (1.0 - a ** 2.0 / b ** 2.0) ** 0.5
        tot_size = scipy.special.ellipeinc(2.0 * np.pi, e)
        arc_size = tot_size / num
        arcs = np.arange(num) * arc_size
        res = scipy.optimize.root(
            lambda x: (scipy.special.ellipeinc(x, e) - arcs), angles)
        angles = res.x
    return angles

def draw(rest, subplot, start, end, drawoffset=0):
    # difference vector
    dist = end - start
    # normalized normal vector
    norm = np.array([-dist[1], dist[0]]) / np.linalg.norm(dist)

    # pos with gather all points
    pos = start
    # the count of the coil
    count = int(rest / 0.1)
    # iterate over them
    for i in range(count):
        # and add a point
        pos = np.vstack(
            (pos, start + dist * (i + 0.5) / count + norm * ((i % 2) * 2 - 1) * 0.1 + norm * drawoffset))
    # add the end point
    pos = np.vstack((pos, end))
    # plot the spring
    subplot.plot(pos[:, 0], pos[:, 1], 'g-')

a = 0.5
b = 2
n = 16

major_axis = b*2
minor_axis = a*2

r = np.sqrt(major_axis/2*minor_axis/2)

r = 1
a = 0.5
b = r**2/a
print("Area", np.pi*a*b)

major_axis = b*2
minor_axis = a*2

x_c = 10
y_c = 20


phi = angles_in_ellipse(160, a, b)
points = np.array([b * np.sin(phi), a * np.cos(phi)])
radii = np.linalg.norm(points, axis=0)
radii[radii < 1] = 1 - radii[radii < 1]
print(radii/r)
print(np.mean(radii/r))
print((major_axis-minor_axis)/np.sqrt(major_axis*minor_axis))
print((b-a)/np.sqrt(a*b))
#exit()

ellipse_angle = 0#/180*np.pi
def plot():
    phi = angles_in_ellipse(n, a, b)
    print(np.round(np.rad2deg(phi), 2))
    # [  0.    16.4   34.12  55.68  90.   124.32 145.88 163.6  180.   196.4 214.12 235.68 270.   304.32 325.88 343.6 ]

    e = (1.0 - a ** 2.0 / b ** 2.0) ** 0.5
    arcs = scipy.special.ellipeinc(phi, e)
    print(np.round(np.diff(arcs), 4))
    # [0.2829 0.2829 0.2829 0.2829 0.2829 0.2829 0.2829 0.2829 0.2829 0.2829 0.2829 0.2829 0.2829 0.2829 0.2829]

    points = np.array([b * np.sin(phi), a * np.cos(phi)])
    alpha = np.deg2rad(ellipse_angle)
    points = points.T @ np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]]) + np.array([x_c, y_c])

    for p in points:
        draw(r, plt.gca(), np.array([x_c, y_c]), p)
    plt.plot(points[:, 0], points[:, 1])
    plt.axis("equal")

#plt.subplot(121)
a = 1
b = 1
n = 8

major_axis = b*2
minor_axis = a*2

r = np.sqrt(major_axis/2*minor_axis/2)
plot()
#plt.subplot(122)
a = 0.5
b = 2

major_axis = b*2
minor_axis = a*2

r = np.sqrt(major_axis/2*minor_axis/2)

x_c = 13
y_c = 20
plot()
plt.show()
