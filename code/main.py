import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import zoom

import plot_utils as pu

class Plane():

    """
    A class to represent either the source or image plane of a gravitational lensing system.
    """

    def __init__(self, 
                 resolution=(101,101),
                 lims=(1,1),
                 fill_value=0.0,
                 rgb=False):

        """
        Initialize the plane with the given resolution and spatial limits.
        :resolution: tuple, (Nx, Ny), the number of pixels in the x and y directions
        :lims: tuple, (x_lim, y_lim), the spatial limits of the plane
        :fill_value: float, the value to fill the plane with
        :rgb: bool, whether the plane is in color or not if you upload an image
        """

        self.Nx, self.Ny = resolution
        self.fill_value = fill_value

        if not rgb:
            self.arr = np.full((self.Nx, self.Ny), fill_value)
        else:
            self.arr = np.full((self.Nx, self.Ny, 3), int(fill_value))

        self.x_lim, self.y_lim = lims

        x = np.linspace(-self.x_lim, self.x_lim, self.Nx)
        y = np.linspace(-self.y_lim, self.y_lim, self.Ny)

        self.px = (2*self.x_lim) / (self.Nx-1)
        self.py = (2*self.y_lim) / (self.Ny-1)

        X, Y = np.meshgrid(x, y)
        self.X, self.Y = X, Y

        self.rgb = rgb

    def add_circle_source(self, center=(0,0), radius=0.1):

        center_x, center_y = center

        dist = np.sqrt((self.X - center_x)**2 + (self.Y - center_y)**2)
        self.arr = np.where(dist <= radius, np.exp(-dist/radius), self.fill_value)
        self.arr /= np.nanmax(self.arr)

    def from_jpeg(self, image_path, lims=None):

        # Load the image
        img = mpimg.imread(image_path)
        x, y = img.shape[:2]
        ratio = x / y

        zoom_x = self.Nx / x
        zoom_y = self.Ny / y

        if self.rgb:
            img_rescaled = zoom(img, (zoom_x, zoom_y, 1), order=1)
        else:
            # Convert to grayscale if not rgb
            img = np.mean(img, axis=2)
            img_rescaled = zoom(img, (zoom_x, zoom_y), order=1) 
    
        self.arr = img_rescaled
        self.arr = self.arr[::-1,::-1]
        
        if lims is None:

            # Redefine the limits taking into consideration the aspect ratio 
            self.x_lim, self.y_lim = 1, 1
            self.y_lim *= ratio

            x = np.linspace(-self.x_lim, self.x_lim, self.Nx)
            y = np.linspace(-self.y_lim, self.y_lim, self.Ny)

            self.px = (2*self.x_lim) / (self.Nx-1)
            self.py = (2*self.y_lim) / (self.Ny-1)

            X, Y = np.meshgrid(x, y)
            self.X, self.Y = X, Y

    def plot(self, cmap='Reds_r', bad_color='k'):

        cmap_name = cmap
        cmap = plt.cm.get_cmap(cmap_name)
        cmap.set_bad(color=bad_color)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        masked_array = np.ma.masked_where(self.arr == 0, self.arr)

        ax.pcolormesh(self.X, self.Y, masked_array, shading='auto', cmap=cmap)

        ax.set_facecolor('black')
        ax.set_aspect('equal')

class GravityLens():

    def __init__(self, 
                 source_plane=None, image_plane=None, 
                 get_magmap=False, get_rays=False):

        if source_plane is None:
            # default source plane is a circle
            plane = Plane()
            plane.add_circle_source()
            self.source_plane = plane
        else:
            self.source_plane = source_plane
        
        if image_plane is None:
            self.image_plane = Plane()
        else:
            self.image_plane = image_plane

        self.point_lenses = []
        self.SIS_lenses = []
        self.CF_lenses = []

        self.rgb = source_plane.rgb
        self.get_magmap = get_magmap
        self.get_rays = get_rays
    
    def add_point_lens(self, position=(0, 0), m=1):

        # Deflection angle for a point mass

        def alpha(x, y):
            x_s, y_s = position
            x_ = x - x_s
            y_ = y - y_s

            r = np.sqrt(x_**2 + y_**2) + 1e-10
            value = m * np.array([x_, y_]) / r**2

            return value

        self.point_lenses.append(alpha)  

    def add_SIS_lens(self, position, theta_E):

        # Deflectin angle for a singular isothermal sphere

        def alpha(x, y):
            x_s, y_s = position
            x_ = x - x_s
            y_ = y - y_s

            r = np.sqrt(x_**2 + y_**2) + 1e-10
            value = theta_E * np.array([x_, y_]) / r

            return value

        self.SIS_lenses.append(alpha)  # Append the new lens function to the list

        return

    # def add_CF_lens(self, position, kappa=0.3, gamma=0.1, m=1):

    #     # Deflection angle for a Vhang-Refsdal lens

    #     def alpha(x, y):
    #         x_s, y_s = position
    #         x_ = x - x_s
    #         y_ = y - y_s

    #         r = np.sqrt(x_**2 + y_**2) + 1e-10

    #         M = np.array([[kappa+gamma, 0], [0, kappa-gamma]])
             
    #         value = np.dot(M, np.array([x, y])) +  m * np.array([x_, y_]) / r**2

    #         return value

    #     self.CF_lenses.append(alpha)

    # add here as many types of lenses as you wish

    def compute_alpha_total(self, x_i, y_i):

        # The total deflectin angle is the sum of the deflection angles from all the lenses

        alpha_total = np.zeros((x_i.shape[0], x_i.shape[1], 2))  

        # Iterate over all lens functions
        for lens_func in self.point_lenses + self.SIS_lenses + self.CF_lenses:
            alpha = np.array(lens_func(x_i, y_i))  # Get alpha from the lens function
            
            # Check and adjust the shape of alpha if necessary
            if alpha.shape != alpha_total.shape:
                # Assuming alpha is (2, height, width), transpose to (height, width, 2)
                alpha = np.transpose(alpha, (1, 2, 0))
            
            alpha_total += alpha

        return alpha_total

    def delete_lenses(self):
        self.point_lenses = []
        self.SIS_lenses = []
        self.CF_lenses = []

    def generate_image(self, save_rays=False, get_magmap=False):

        # only if you want to copmute the positions of the rays in the optical axis plane
        get_rays = self.get_rays or save_rays
        if get_rays:
            rays = []

        # if you want to compute the magnification map
        get_magmap = self.get_magmap or get_magmap

        if get_magmap:
            mag_map = np.zeros((self.source_plane.Nx, self.source_plane.Ny))

        x_i, y_i = self.image_plane.X, self.image_plane.Y
        alpha_total = self.compute_alpha_total(x_i, y_i)

        x_s = x_i - alpha_total[:,:,0]
        y_s = y_i - alpha_total[:,:,1]

        # Calculate source indices
        i_s = np.round(((self.source_plane.Nx-1) * (x_s + self.source_plane.x_lim)) / (2 * self.source_plane.x_lim)).astype(int)
        j_s = np.round(((self.source_plane.Ny-1) * (y_s + self.source_plane.y_lim)) / (2 * self.source_plane.y_lim)).astype(int)

        # Check if indices are within the valid range
        within_range_x = (i_s >= 0) & (i_s < self.source_plane.Nx)
        within_range_y = (j_s >= 0) & (j_s < self.source_plane.Ny)
        within_range = within_range_x & within_range_y

        # Update image plane
        valid_i_i, valid_j_i = np.where(within_range)
        for index in range(len(valid_i_i)):
            i_i, j_i = valid_i_i[index], valid_j_i[index]
            if self.rgb:
                self.image_plane.arr[i_i, j_i, :] = self.source_plane.arr[i_s[i_i, j_i], j_s[i_i, j_i], :]
            else:
                self.image_plane.arr[i_i, j_i] = self.source_plane.arr[i_s[i_i, j_i], j_s[i_i, j_i]]

            if get_rays:
                rays.append([(x_i[i_i, j_i], y_i[i_i, j_i]), (x_s[i_i, j_i], y_s[i_i, j_i])])
                                                        
            if get_magmap:
                mag_map[i_s[i_i, j_i], j_s[i_i, j_i]] += 1

        if get_rays:
            self.rays = np.array(rays)
        
        if get_magmap:
            self.magmap = mag_map.T

    def plot_image(self, cmap='binary_r', savepath=None, fig_size=720):
        Fig = pu.Figure(subplots=(1,1), ratio=1, fig_size=fig_size, grid=False)
        axs = Fig.axes_flat

        for ax in axs:
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

        if not self.rgb:
            image_masked_array = np.ma.masked_where(self.image_plane.arr == 0, self.image_plane.arr)
            image_masked_array = image_masked_array.T
            axs[0].pcolormesh(self.image_plane.X, self.image_plane.Y, image_masked_array, shading='auto', cmap=cmap)

        else:

            im = self.image_plane.arr

            im = np.transpose(im, (1, 0, 2))

            axs[0].imshow(im, extent=[-1, 1, -1, 1])

        axs[0].set_aspect('equal')

        plt.tight_layout(pad=0)

        # Save the figure without padding and in the original dimension
        if savepath != None:
            plt.savefig(savepath, bbox_inches='tight', pad_inches=0, dpi=300)

            plt.close()

    def plot_source(self, cmap='binary_r', savepath=None, fig_size=720):
        Fig = pu.Figure(subplots=(1,1), ratio=1, fig_size=fig_size, grid=False)
        axs = Fig.axes_flat

        for ax in axs:
            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.set_xticks([])
            ax.set_yticks([])

        if not self.rgb:
            source_masked_array = np.ma.masked_where(self.source_plane.arr == 0, self.source_plane.arr)
            source_masked_array = source_masked_array.T
            axs[0].pcolormesh(self.source_plane.X, self.source_plane.Y, source_masked_array, shading='auto', cmap=cmap)

        else:

            im = self.source_plane.arr

            im = np.transpose(im, (1, 0, 2))

            axs[0].imshow(im, extent=[-self.source_plane.x_lim,
                                      self.source_plane.x_lim,
                                      -self.source_plane.y_lim,
                                      self.source_plane.y_lim])

        axs[0].set_aspect('equal')

        plt.tight_layout(pad=0)

        # Save the figure without padding and in the original dimension
        if savepath != None:
            plt.savefig(savepath, bbox_inches='tight', pad_inches=0, dpi=300)

            plt.close()

    def plot_source_image(self, cmap='Reds_r', bad_color='k', wspace=0.1):

        Fig = pu.Figure(subplots=(1,2), ratio=1, fig_size=720, wspace=wspace, grid=False)

        axs = Fig.axes_flat

        if not self.rgb:

            cmap = plt.cm.get_cmap(cmap)
            cmap.set_bad(color=bad_color)
            source_masked_array = np.ma.masked_where(self.source_plane.arr == 0, self.source_plane.arr)
            image_masked_array = np.ma.masked_where(self.image_plane.arr == 0, self.image_plane.arr)
            image_masked_array = image_masked_array.T

            axs[0].pcolormesh(self.source_plane.X, self.source_plane.Y, source_masked_array, shading='auto', cmap=cmap)
            axs[1].pcolormesh(self.image_plane.X, self.image_plane.Y, image_masked_array, shading='auto', cmap=cmap)

        else:
            axs[0].imshow(self.source_plane.arr, extent=[-1,1,-1,1])
            axs[1].imshow(self.image_plane.arr, extent=[-1,1,-1,1])

        Fig.customize_axes(axs[1], ylabel_pos='right')

        for ax in axs:
            ax.set_aspect('equal')
            

    def plot_magmap(self, cmap='jet', bad_color='k', savepath=None, fig_size=720, vmax=None):
        Fig = pu.Figure(subplots=(1,1), ratio=1, fig_size=fig_size, grid=False, color='k')
        axs = Fig.axes_flat

        for ax in axs:
            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.set_xticks([])
            ax.set_yticks([])


        masked_array = np.ma.masked_where(self.magmap == 0, self.magmap)
        masked_array = masked_array.T

        axs[0].pcolormesh(self.source_plane.X, self.source_plane.Y, np.log(masked_array+0.01), shading='auto', cmap=cmap, vmax=vmax)

        axs[0].set_aspect('equal')

        plt.tight_layout(pad=0)

        # Save the figure without padding and in the original dimension
        if savepath != None:
            plt.savefig(savepath, bbox_inches='tight', pad_inches=0, dpi=300)

            plt.close()

