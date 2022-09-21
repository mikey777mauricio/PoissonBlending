import notebook as notebook
import numpy as np  # for building and manipulating matrices
import time  # for measuring time elapsed while running code

# for sparse matrix operations
from matplotlib_inline.config import InlineBackend
from scipy.sparse import lil_matrix
import scipy.sparse.linalg as sla
from skimage.transform import resize

# for graphics
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
notebook
InlineBackend.figure_format = 'retina'  # nice retina graphics


# vertical constraint
def add_vertical_constraints(A, b, j, counter, im2var, imh, im_s, im_background):
    # loop through rows
    for i in range(0, imh - 1):
        # first boundary case
        if i == 0:

            A[counter, im2var[i + 1, j]] = 1
            b[counter] = im_s[i + 1, j] - im_s[i, j] + im_background[i, j]

        # second boundary case
        elif i == imh - 2:

            A[counter, im2var[i, j]] = -1
            b[counter] = im_s[i + 1, j] - im_s[i, j] - im_background[i + 1, j]

        # normal
        else:

            A[counter, im2var[i, j]] = -1
            A[counter, im2var[i + 1, j]] = 1
            b[counter] = im_s[i + 1, j] - im_s[i, j]
        # increment counter
        counter += 1
    # return counter for horizontal constraint
    return counter


# horizontal constraint
def add_horizontal_constraints(A, b, i, counter, im2var, imw, im_s, im_background):
    # loop through columns
    for j in range(0, imw - 1):
        # first boundary case
        if j == 0:
            A[counter, im2var[i, j + 1]] = 1
            b[counter] = im_s[i, j + 1] - im_s[i, j] + im_background[i, j]

        # second boundary case
        elif j == imw - 2:

            A[counter, im2var[i, j]] = -1
            b[counter] = im_s[i, j + 1] - im_s[i, j] - im_background[i, j + 1]

        # normal
        else:
            A[counter, im2var[i, j]] = -1
            A[counter, im2var[i, j + 1]] = 1
            b[counter] = im_s[i, j + 1] - im_s[i, j]

        # increment counter
        counter += 1
    # return
    return counter


def least_squares_2D(im_s, im_background):
    #### initialize results to just be copies of the background
    im_blend = im_background.copy()

    #### initialize the size of source image
    imh, imw = (im_s.shape[0], im_s.shape[1])  #### imh: image hieght, imw: image width
    imd = 1 if im_s.ndim < 3 else im_s.shape[2]  #### number of components of each pixel, e.g., for an RGB image, imd=3

    #### TODO 1: specify the number of unknowns and number of equations ####
    # Hint: num_vars = ?
    # Hint: num_eqns = ?
    #### TODO END ####
    # unknowns = imh-2 * imw-2
    # eqns = # of edges

    num_vars = (imh - 2) * (imw - 2)
    num_eqns = (imw - 2) * (imh - 1) + (imh - 2) * (imw - 1)

    #### TODO 2: initialize the vectorization mapping ####
    # Hint: You need to create an im2var matrix with the same dimensions as im_s
    # and assign coordinates inside the 1 pixel padding unique numbers for easy indexing.
    # Hint: im2var = ?
    #### TODO END ####
    im2var = np.zeros((im_s.shape[0], im_s.shape[1]))

    # set 1 pixel padding
    for i in range(0, imh):
        im2var[i, 0] = -1
        im2var[i, im_s.shape[1] - 1] = -1

    for i in range(0, imw):
        im2var[0][i] = -1
        im2var[im_s.shape[0] - 1, i] = -1

    # k value for indexing
    k = 0
    for i in range(1, im_s.shape[0] - 1):
        for j in range(1, im_s.shape[1] - 1):
            # set index
            im2var[i, j] = k
            # increment
            k += 1

    #### check if the mapping is initialized correctly
    print(im2var)

    #### initialize A and b ####
    A = lil_matrix((num_eqns, num_vars))

    b = np.zeros((num_eqns, imd))

    #### TODO 3: fill the elements of A and b ####
    #### Hint: This is the major step of this assignment.
    #### Hint: Initialize all the vertical constraints first and then all the horizontal constraints
    #### Hint: Write a for loop to go through all the columns (for vertical constraints) and all the rows (for horizontal ones)
    #### Hint: For each column, consider three cases: upper boundary, interior, and lower boundary
    #### Hint: For each boundary case, consider how to initialize b
    #### Hint: Here you don't need to initialize each component of b separately. Instead, do it as a whole like b[e]=...
    #### TODO END ####

    # A and B

    # initialize counter
    counter = 0

    # vertical
    for j in range(1, imw - 1):
        counter = add_vertical_constraints(A, b, j, counter, im2var, imh, im_s, im_background)

    # horizontal
    for i in range(1, imh - 1):
        counter = add_horizontal_constraints(A, b, i, counter, im2var, imw, im_s, im_background)

    #### solve the least-squares problem
    x = sla.lsqr(A, b[:, imd - 1])[0]

    #### convert A to (CSR) sparse format
    A = A.tocsr()

    print('Solving sparse system using sla.lsqr...')
    t = time.time()

    #### solve for all channels
    for c in range(0, imd):
        #### TODO 4: solve the least-squares problem Av=b, with the right-hand side specified by the c'th column of b ####
        #### Hint: This is just one line of code
        #### Hint: Use the function 'sla.lsqr' to solve the least-squares problem for efficiency.
        #### Hint: 'sla.lsqr' reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html
        #### Hint: Don't forget to specify the column index when you assign the results to v.
        #### TODO END ####

        # solve for c column of b
        v = sla.lsqr(A, b[:, c])[0]

        #### copy over to im_blend
        if im_s.ndim > 2:  #### this branch is for rgb image
            #### TODO 5: reshape v to an appropriate shape and copy its values to the c'th component of im_blend ####
            #### Hint: This is just one line of code
            #### Hint: Consider using the function reshape.
            #### Hint: Be careful with the dimension
            #### TODO END ####

            # copy values into c component
            im_blend[1:imh - 1, 1:imw - 1, c] = v.reshape(im_blend.shape[0] - 2, im_blend.shape[1] - 2)

        else:  #### this branch is for gray-scale image
            #### TODO 5: reshape v to an appropriate shape and copy its values to im_blend ####
            #### Hint: This is just one line of code
            #### Hint: The same above. For an gray-scale image, you don't need to specify the third dimension (c)
            #### TODO END ####

            # copy values
            im_blend[1:imh - 1, 1:imw - 1] = v.reshape(im_blend.shape[0] - 2, im_blend.shape[1] - 2)

    elapsed = time.time() - t
    print("\tTotal time for sparse solve: {} seconds \n".format(elapsed))

    #### return the blended result
    return im_blend

## new gradient domain copy-paste code should go here
# read in background and object images
im_background = plt.imread('images/background.jpg')/255.0
im_object = plt.imread('images/penguin-chick.jpg')/255.0
#
# reduce the size of the image for faster performance while debugging
# You can comment these two lines out to use the full-resolution images
im_background = resize(im_background, (im_background.shape[0] // 5,
                                       im_background.shape[1] // 5),
                       anti_aliasing=True)
im_object = resize(im_object, (im_object.shape[0] // 5,
                               im_object.shape[1] // 5),
                       anti_aliasing=True)

# get source region mask from the user
objh, objw, _ = im_object.shape
objmask = np.ones((objh, objw))

# for storing the selected coordinates aligning blending
coords = np.zeros(2)

# handling coordinate selection by storing selected coordinates
def onclick(event):
    global coords, objh, objw
    coords = np.round([event.xdata, event.ydata]).astype(np.int)
    plt.title(f"Background Image (Selected Coordinate {coords[0]}, {coords[1]})")
    # draw rectangle on image
    rect = Rectangle((event.xdata-objw,event.ydata-objh),objw,objh,linewidth=1,edgecolor='r',facecolor='none')
    plt.gca().add_patch(rect)

# display interactive figure for selecting coordinate in background image
fig = plt.figure()
fig.set_size_inches(10,8)
plt.imshow(im_background)
cid = fig.canvas.mpl_connect('button_press_event', onclick) # for handling button click
plt.title("Background Image (Click a place to blend object image)")
plt.show()


def simple_copy_paste(coords, im_object, im_background):
    x, y = coords
    objh, objw, _ = im_object.shape

    # paste pixel values into im_background
    result = im_background.copy()
    result[y - objh:y, x - objw:x, :] = im_object

    return result


def poisson_copy_paste(coords, im_object, im_background):
    x, y = coords
    objh, objw, _ = im_object.shape

    res = im_background.copy()
    background = im_background[y - objh:y, x - objw:x, :]

    #### TODO: cut out the appropriate region of im_background and paste back into im_background ####
    #### TODO END ####

    # solve for im_blend
    im_blend = least_squares_2D(im_object, background)

    # paste result onto background
    res = simple_copy_paste(coords, im_blend, res)

    res0 = simple_copy_paste(coords, im_object, im_background)

    fig3, axs3 = plt.subplots(1, 2)
    fig3.tight_layout(pad=0.0)
    fig3.set_size_inches(14, 5)

    # show simple copy-paste first
    axs3[0].imshow(res0.clip(0, 1))
    axs3[0].set_title("Simple")

    # TODO: your code here
    # add two more sub-plots with your poisson copy-paste results
    res = poisson_copy_paste(coords, im_object, im_background)

    # show poisson copy-paste
    axs3[1].imshow(res.clip(0, 1))
    axs3[1].set_title("Poisson")

    ### TODO END
    plt.show()
