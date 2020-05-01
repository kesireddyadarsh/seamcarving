import numpy as np
import cv2
import scipy.misc
import scipy.ndimage
from PIL import Image as PILImage
import imageio
import moviepy.editor as mpy
from matplotlib import pyplot as plt


class AnimationMaker:
    """
    Makes an animation out of a sequence of Images
    """
    def __init__(self):
        self.sequence = []
        self.max_width = 0
        self.max_height = 0
        self.max_dimensions = 0
        self.default_FPS = 25
        self.current_step = 0

    def add(self, d_image):
        self.sequence.append(d_image)
        if self.max_height <= d_image.height:
            self.max_height = d_image.height
        if self.max_width <= d_image.width:
            self.max_width = d_image.width
        if self.max_dimensions <= d_image.dim:
            self.max_dimensions = d_image.dim


class Image:
    def __init__(self, array=None, transposed=False):
        #print("This is in __init__ function  from Image\n\n")
        self._array = array
        self.greyscale_coeffs = [.299, .587, .144]
        self.transposed = transposed
        self.greyscale_image = None
        self.sobel_image = None
        self.canny = None
        self.min_energy_image = None

    @property
    def array(self):
        """
        :return: the image array (transposed if needed)
        """
        #print("This is in array function  from Image\n\n")
        if self.transposed:
            if self.dim == 3:
                return self._array.transpose(1, 0, 2)
            else:
                return self._array.transpose(1, 0)
        return self._array

    @property
    def width(self):
        #print("This is in width function  from Image\n\n")
        if self.transposed:
            return self._array.shape[0]
        return self._array.shape[1]

    @property
    def height(self):
        if self.transposed:
            return self._array.shape[1]
        return self._array.shape[0]

    @property
    def dim(self):
        if len(self._array.shape) > 2:
            return self._array.shape[2]
        return 2

    @classmethod
    def from_image(cls, image):
        return cls(image.array, image.transposed)

    @classmethod
    def from_image_array(cls, array, transposed=False):
        return cls(array, transposed)

    @classmethod
    def from_file(cls, image_file):
        return cls(imageio.imread(image_file))

    @property
    def greyscale(self):
        """
        :return: greyscale image transposed if needed
        """
        #print("This is in greyscale function  from Image\n\n")
        if not self.greyscale_image:
            self.greyscale_image = np.dot(self.array[:, :, :3], self.greyscale_coeffs)
        return self.greyscale_image


    @property
    def energy(self):
        #print("This is in energy function  from Image\n\n")
        if not self.sobel_image:
            greyscale = self.greyscale.astype('int32')
            dx = scipy.ndimage.sobel(greyscale, 0)  # horizontal derivative
            dy = scipy.ndimage.sobel(greyscale, 1)  # vertical derivative
            self.sobel_image = np.hypot(dx, dy)  # magnitude
            self.sobel_image *= 255.0 / np.max(self.sobel_image)  # normalize
        return self.sobel_image

    @property
    def min_energy(self):
        """
        Converts energy values to cumulative energy values
        """
        #print("This is in min_energy function  from Image\n\n")
        if not self.min_energy_image:
            image = self.energy
            self.min_energy_image = np.zeros((self.height, self.width))
            self.min_energy_image[0][:] = image[0][:]
            for i in range(self.height):
                for j in range(self.width):
                    if i == 0:
                        #print("Fist loop")
                        self.min_energy_image[i, j] = image[i, j]
                    elif j == 0:
                        self.min_energy_image[i, j] = image[i, j] + min(
                            self.min_energy_image[i - 1, j],
                            self.min_energy_image[i - 1, j + 1]
                        )
                    elif j == self.width - 1:
                        #print("second condtion")
                        self.min_energy_image[i, j] = image[i, j] + min(
                            self.min_energy_image[i - 1, j - 1],
                            self.min_energy_image[i - 1, j]
                        )
                    else:
                        #print("Third condition")
                        self.min_energy_image[i, j] = image[i, j] + min(
                            self.min_energy_image[i - 1, j - 1],
                            self.min_energy_image[i - 1, j],
                            self.min_energy_image[i - 1, j + 1]
                        )

        return self.min_energy_image

    def debug(self, seam):
        """
        :param seam: current seam in the image (2 dim array with one column/row)
        :return: a debug image showing the actual image being processed with the currently chosen seam
        """
        #print("This is in # DEBUG:  function  from Image\n\n")
        image = self.array
        color = [255] * 3
        seam_array = seam.array
        size = seam.width if seam.width > seam.height else seam.height
        #print(color)
        for i in range(size):
            if seam.transposed:
                image[i][seam_array[0][i]] = color
            else:
                image[i][seam_array[i][0]] = color
        #print(color)
        return image

    def save(self, filename):
        #print("This is in save function  from Image\n\n")
        im = PILImage.fromarray(self.array.astype('uint8'))
        im.save(filename)


class SeamCarver:
    def __init__(self, image_file):
        #print("THis is in SeamCarver \n\n\n")
        self.image = Image.from_file(image_file)
        self.debug_animation = AnimationMaker()


    def seams(self, n):
        """
        Finds the n minimum seams in the image
        :param n: the number of paths to be found
        :return a list of Image objects, each containing the a path, sorted ascending
        """
        #print("This is in Seams function from SeamCarver\n\n")
        d_image = self.image.min_energy
        #print(d_image)
        #print(self.image)
        #print(self.image.min_energy)
        #print("Debugging\n\n")
        #print(self.image)
        seams_found = []
        for iteration in range(n):
            seam_image = np.zeros((self.image.height, 1), dtype=int)
            #print(seam_image)
            #print(len(d_image))
            #print(seams_found)
            for i in reversed(range(0, self.image.height)):
                if i == self.image.height - 1:
                    value = min(d_image[i, :])
                    j = np.where(d_image[i][:] == value)[0][0]
                else:
                    if seam_image[i + 1, 0] == 0:
                        tmp = [
                            float("Inf"),
                            d_image[i, seam_image[i + 1, 0]],
                            d_image[i, seam_image[i + 1, 0] + 1]
                        ]
                    elif seam_image[i + 1, 0] == self.image.width - 1:
                        tmp = [
                            d_image[i, seam_image[i + 1, 0] - 1],
                            d_image[i, seam_image[i + 1, 0]],
                            float("Inf")
                        ]
                    else:
                        #print(image[i,1])
                        #print(seam_image)
                        tmp = [
                            d_image[i, seam_image[i + 1, 0] - 1],
                            d_image[i, seam_image[i + 1, 0]],
                            d_image[i, seam_image[i + 1, 0] + 1]
                        ]
                    j = seam_image[i+1, 0] + np.argmin(tmp) - 1
                seam_image[i, 0] = j
                d_image[i, j] = float("Inf")
                #print(Image.from_image_array)
            seams_found.append(Image.from_image_array(array=seam_image, transposed=self.image.transposed))
            #print(Image.from_image_array)
            #print(len(d_image))
            #print(seams_found)
            #print(seam_image)
        return seams_found

    def cut_seam(self):
        #print("This is in cut_seam function \n\n")
        seam = self.seams(1)[0]
        output = np.zeros((self.image.height, self.image.width - 1, self.image.dim))
        #print(seam.transposed)
        for i in range(0, self.image.dim):
            for j in range(0, self.image.height):
                if seam.transposed:
                    #print(j)
                    output[j, :, i] = np.append(self.image.array[j, 0: seam.array[0, j], i],
                                                self.image.array[j, seam.array[0, j] + 1: self.image.width, i]
                                                )
                else:
                    output[j, :, i] = np.append(self.image.array[j, 0: seam.array[j, 0], i],
                                                self.image.array[j, seam.array[j, 0] + 1: self.image.width, i]
                                                )
        temp_image = self.image.debug(seam)
        #print(temp_image)
        self.debug_animation.add(Image.from_image_array(temp_image, transposed=self.image.transposed))

        if self.image.transposed:
            output = output.transpose(1, 0, 2)
        #print(len(output))

        return Image.from_image_array(array=output, transposed=self.image.transposed)

    def add_seam(self, seam):
        #print("This is for add_seam function \n\n")
        output = np.zeros((self.image.height, self.image.width + 1, self.image.dim))
        for i in range(0, self.image.dim):
            for j in range(0, self.image.height):
                x = seam.array[0, j] if seam.transposed else seam.array[j, 0]
                if x < self.image.width - 2:
                    tmp_vector_avg = np.array([(self.image.array[j, x, i] + self.image.array[j, x + 1, i])/2.0])
                else:
                    tmp_vector_avg = np.array([(self.image.array[j, x, i] + self.image.array[j, x - 1, i])/2.0])

                tmp = np.append(self.image.array[j, 0: x + 1, i], tmp_vector_avg)
                output[j, :, i] = np.append(tmp, self.image.array[j, x + 1: self.image.width, i])

        temp_image = self.image.debug(seam)
        self.debug_animation.add(Image.from_image_array(temp_image, transposed=self.image.transposed))

        if self.image.transposed:
            output = output.transpose(1, 0, 2)

        return Image.from_image_array(array=output, transposed=seam.transposed)

    def cut_seams(self, desired_w):
        generations = self.image.width - desired_w
        #print("generations")
        #print(generations)
        current_generation = 0
        while current_generation < generations:
            #print("In while loop cut_seams")
            #plt.show()
            self.image = self.cut_seam()
            current_generation += 1

    def add_seams(self, desired_w):
        generations = desired_w - self.image.width
        seams = self.seams(generations)
        current_generation = 0
        while current_generation < generations:
            #print("In while loop add_seams")
            self.image = self.add_seam(seams[current_generation])
            current_generation += 1

    def resize(self, desired_w, desired_h):
        if desired_w < self.image.width:
            self.cut_seams(desired_w)
        else:
            self.add_seams(desired_w)

        if desired_h < self.image.height:
            self.image.transposed = True
            self.cut_seams(desired_h)
            self.image.transposed = False
        else:
            self.image.transposed = True
            self.add_seams(desired_h)
            self.image.transposed = False

        return self.image


def plothistogram(Path_to_file,save_to_file):
    #fig=plt.figure(figsize=(1, 2))
    img = cv2.imread(Path_to_file)
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    #plt.savefig("/Users/adarshkesireddy/Documents/Research/Coding/AnalysisProject/main_image.jpg")
        #fig.add_subplot(1,1)
        #plt.imshow(histr)
    #plt.show()
    img_2 = cv2.imread(save_to_file)
    color_2 = ('b','g','r')
    for i,col_2 in enumerate(color_2):
        histr_2 = cv2.calcHist([img_2],[i],None,[256],[0,256])
        plt.plot(histr_2,color = col_2)
        plt.xlim([0,256])
    plt.savefig("/Users/adarshkesireddy/Documents/Research/Coding/AnalysisProject/smaller_image_1.jpg")
        #fig.add_subplot(1,2)
        #plt.imshow(histr)
    #plt.show()

Path_to_file = "/Users/adarshkesireddy/Documents/Research/Coding/AnalysisProject/004.jpg"
save_to_file = "/Users/adarshkesireddy/Documents/Research/Coding/AnalysisProject/smaller_1.jpg"


if __name__ == '__main__':
    #scale_down_example()
    sc = SeamCarver(Path_to_file)
    image = sc.resize(394, 499)
    #print(len(image))
    image.save(save_to_file)
    plothistogram(Path_to_file,save_to_file)
    #plothistogram(save_to_file)
