from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7})


###CLASS APPLICATION
class application:
    def __init__(self, master):
        self.master = master
        self.gui_size = (600, 400)
        self.set_gui()
        self.height = 400
        # self.master.resizable(0,0)


    def open_image(self):
        self.img0 = Image.open(self.dir)


    def load_image(self):
        self.dir = self.entry_img.get()
        self.open_image()
        self.show_image1()


    def browse_image(self):
        self.dir = filedialog.askopenfilename()
        self.open_image()
        self.show_image1()


    def show_image1(self):
        new_size = int((float(self.img0.size[0])*float(self.height / float(self.img0.size[1]))))
        self.img = self.img0.resize((new_size, self.height), Image.ANTIALIAS)
        self.img_cache = self.img
        self.img1 = ImageTk.PhotoImage(self.img)
        self.canvas_img.delete(ALL)
        self.canvas_img.create_image(self.img.size[0]/2, self.img.size[1]/2, anchor=CENTER, image=self.img1)
        self.status['text'] = 'Image: ' + self.dir
        self.entry_img.delete(0, END)
        self.entry_img.insert(0, self.dir)


    def show_image2(self):
        self.img2 = ImageTk.PhotoImage(self.img)
        self.canvas_img.delete(ALL)
        self.canvas_img.create_image(self.img.size[0]/2, self.img.size[1]/2, anchor=CENTER, image=self.img2)


    def greyscale_img(self):
        img_bn = Image.new("RGB", (int(self.img.size[0]), int(self.img.size[1])))
        for x in range(0, self.img.size[0]):
            for y in range(0, self.img.size[1]):
                r, g, b = self.img.getpixel((x, y))
                img_bn.putpixel((x, y), (int((r+g+b)/3),int((r+g+b)/3),int((r+g+b)/3)))
        self.img = img_bn
        self.show_image2()

    def grayscale_img(self):
        array_gambar = np.array(self.img, dtype=float)
        new_image = np.zeros((array_gambar.shape[0], array_gambar.shape[1]), dtype=np.int)
        for x in range(array_gambar.shape[0]):
            for y in range(array_gambar.shape[1]):
                temp = array_gambar[x, y, 0] + array_gambar[x, y, 1] + array_gambar[x, y, 2]
                temp /= 3
                new_image[x, y] = temp

        new_image = np.clip(new_image, 0, 255)
        new_image = np.array(new_image, dtype=np.uint8)
        new_image = Image.fromarray(new_image, "L")
        self.img = new_image
        self.show_image2()


    def zoom_in_img(self):
        img_zi = Image.new("RGB", (self.img.size[0]*2, self.img.size[1]*2))
        for x in range(0, img_zi.size[0], 2):
            for y in range(0, img_zi.size[1], 2):
                r, g, b = self.img.getpixel((int(x/2), int(y/2)))
                img_zi.putpixel((x, y), (r, g, b))
                img_zi.putpixel((x+1, y), (r, g, b))
                img_zi.putpixel((x, y+1), (r, g, b))
                img_zi.putpixel((x+1, y+1), (r, g, b))
        self.img = img_zi
        self.show_image2()




    def zoom_out_img(self):
        img_zo = Image.new("RGB", (int(self.img.size[0]/2), int(self.img.size[1]/2)))
        for x in range(0, self.img.size[0], 2):
            for y in range(0, self.img.size[1], 2):
                r, g, b = self.img.getpixel((x+1, y))
                r1, g1, b1 = self.img.getpixel((x+1, y))
                r2, g2, b2 = self.img.getpixel((x, y+1))
                r3, g3, b3 = self.img.getpixel((x+1, y+1))
                img_zo.putpixel((int(x/2), int(y/2)), (int((r+r1+r2+r3)/4), int((g+g1+g2+g3)/4), int((b+b1+b2+b3)/4)))
        self.img = img_zo
        self.show_image2()


    def inc_brightness_plus(self):
        # array_baru = np.array(self.img)
        # array_baru += 10
        # img_bn = Image.fromarray(array_baru)
        # self.img = img_bn
        # self.show_image2()

        img_bn = Image.new("RGB", (int(self.img.size[0]), int(self.img.size[1])))
        for x in range(0, self.img.size[0]):
            for y in range(0, self.img.size[1]):
                r, g, b = self.img.getpixel((x, y))
                img_bn.putpixel((x, y), (r+10, g+10, b+10))
        self.img = img_bn
        self.show_image2()


    def dec_brightness_min(self):
        # array_baru = np.array(self.img)
        # array_baru -= 10
        # img_bn = Image.fromarray(array_baru)
        # self.img = img_bn
        # self.show_image2()

        img_bn = Image.new("RGB", (int(self.img.size[0]), int(self.img.size[1])))
        for x in range(0, self.img.size[0]):
            for y in range(0, self.img.size[1]):
                r, g, b = self.img.getpixel((x, y))
                img_bn.putpixel((x, y), (r - 10, g - 10, b - 10))
        self.img = img_bn
        self.show_image2()


    def inc_brightness_times(self):
        # array_baru = np.array(self.img)
        # array_gambar = np.array(array_baru * 1.1, dtype='uint8')
        # img_bn = Image.fromarray(array_gambar)
        # self.img = img_bn
        # self.show_image2()

        img_bn = Image.new("RGB", (int(self.img.size[0]), int(self.img.size[1])))
        for x in range(0, self.img.size[0]):
            for y in range(0, self.img.size[1]):
                r, g, b = self.img.getpixel((x, y))
                img_bn.putpixel((x, y), (int(r * 1.1), int(g * 1.1), int(b * 1.1)))
        self.img = img_bn
        self.show_image2()


    def dec_brightness_by(self):
        array_baru = np.array(self.img)
        array_gambar = np.array(array_baru/1.1, dtype='uint8')
        img_bn = Image.fromarray(array_gambar)
        self.img = img_bn
        self.show_image2()

        # img_bn = Image.new("RGB", (int(self.img.size[0]), int(self.img.size[1])))
        # for x in range(0, self.img.size[0]):
        #     for y in range(0, self.img.size[1]):
        #         r, g, b = self.img.getpixel((x, y))
        #         img_bn.putpixel((x, y), (int(r/1.2), int(g/1.2), int(b/1.2)))
        # self.img = img_bn
        # self.show_image2()


    def move_up(self):
        array_gambar = np.array(self.img)
        array_baru = np.zeros((array_gambar.shape[0], array_gambar.shape[1], array_gambar.shape[2]), dtype='uint8')
        array_baru[:-10, :, :] += array_gambar[10:, :, :]
        img_mu = Image.fromarray(array_baru)
        self.img = img_mu
        self.show_image2()


    def move_down(self):
        array_gambar = np.array(self.img)
        array_baru = np.zeros((array_gambar.shape[0], array_gambar.shape[1], array_gambar.shape[2]), dtype='uint8')
        array_baru[10:, :, :] += array_gambar[:-10, :, :]
        img_mu = Image.fromarray(array_baru)
        self.img = img_mu
        self.show_image2()


    def move_left(self):
        array_gambar = np.array(self.img)
        array_baru = np.zeros((array_gambar.shape[0], array_gambar.shape[1], array_gambar.shape[2]), dtype='uint8')
        array_baru[:, :-10, :] += array_gambar[:, 10:, :]
        img_mu = Image.fromarray(array_baru)
        self.img = img_mu
        self.show_image2()


    def move_right(self):
        array_gambar = np.array(self.img)
        array_baru = np.zeros((array_gambar.shape[0], array_gambar.shape[1], array_gambar.shape[2]), dtype='uint8')
        array_baru[:, 10:, :] += array_gambar[:, :-10, :]
        img_mu = Image.fromarray(array_baru)
        self.img = img_mu
        self.show_image2()


    def histogram_it(self):
        array_gambar = np.array(self.img)

        self.r = array_gambar[:, :, 0]
        self.g = array_gambar[:, :, 1]
        self.b = array_gambar[:, :, 2]

        # self.rkey, self.rcounts = np.unique(r, return_counts=True)
        # self.gkey, self.gcounts = np.unique(g, return_counts=True)
        # self.bkey, self.bcounts = np.unique(b, return_counts=True)

        self.rhist = [0]*256
        for i in range(self.r.shape[0]):
            for j in range(self.r.shape[1]):
                self.rhist[self.r[i, j]]+= 1

        self.ghist = [0] * 256
        for i in range(self.g.shape[0]):
            for j in range(self.g.shape[1]):
                self.ghist[self.g[i, j]] += 1

        self.bhist = [0] * 256
        for i in range(self.b.shape[0]):
            for j in range(self.b.shape[1]):
                self.bhist[self.b[i, j]] += 1

        self.rhistogram.cla()
        self.ghistogram.cla()
        self.bhistogram.cla()

        self.rhistogram.set_title('Red', fontsize=9)
        self.rhistogram.set_xlabel('Intensity', fontsize=6)
        self.rhistogram.set_ylabel('Total Intensity', fontsize=6)

        self.ghistogram.set_title('Green', fontsize=9)
        self.ghistogram.set_xlabel('Intensity', fontsize=6)
        self.ghistogram.set_ylabel('Total Intensity', fontsize=6)

        self.bhistogram.set_title('Blue', fontsize=9)
        self.bhistogram.set_xlabel('Intensity', fontsize=6)
        self.bhistogram.set_ylabel('Total Intensity', fontsize=6)

        # self.rhistogram.bar(self.rkey, self.rcounts)
        # self.ghistogram.bar(self.gkey, self.gcounts)
        # self.bhistogram.bar(self.bkey, self.bcounts)

        self.rhistogram.bar(range(256), self.rhist)
        self.ghistogram.bar(range(256), self.ghist)
        self.bhistogram.bar(range(256), self.bhist)

        self.show_histogram()


    def histeq_it(self):
        array_gambar = np.array(self.img)

        self.r = array_gambar[:, :, 0]
        self.g = array_gambar[:, :, 1]
        self.b = array_gambar[:, :, 2]

        rcdf = [0] * 256
        rcdf[0] = self.rhist[0]
        for i in range(1, 256):
            rcdf[i] = rcdf[i - 1] + self.rhist[i]
        rcdf = [i * 255 / rcdf[-1] for i in rcdf]
        rout = np.interp(self.r, range(256), rcdf)

        gcdf = [0] * 256
        gcdf[0] = self.ghist[0]
        for i in range(1, 256):
            gcdf[i] = gcdf[i - 1] + self.ghist[i]
        gcdf = [i * 255 / gcdf[-1] for i in gcdf]
        gout = np.interp(self.g, range(256), gcdf)

        bcdf = [0] * 256
        bcdf[0] = self.bhist[0]
        for i in range(1, 256):
            bcdf[i] = bcdf[i - 1] + self.bhist[i]
        bcdf = [i * 255 / bcdf[-1] for i in bcdf]
        bout = np.interp(self.b, range(256), bcdf)

        new_image_array = np.dstack((rout, gout, bout)).astype(np.uint8)
        new_image = Image.fromarray(new_image_array,"RGB")
        self.img = new_image
        self.show_image2()
        self.histogram_it()

    def show_histogram(self):
        self.histogram_canvas.draw()
        self.histogram_canvas.get_tk_widget().pack(side=RIGHT, expand=True)

    def index_check(self,x):
        try:
            data = x
        except IndexError:
            data = 0
        return data


    def edge_detect(self):
        array_gambar = np.array(self.img, dtype=np.float)

        new_image = np.zeros((array_gambar.shape[0], array_gambar.shape[1], array_gambar.shape[2]), dtype=np.uint8)

        for i in range(array_gambar.shape[0]-1):
            for j in range(array_gambar.shape[1]-1):
                temp = self.index_check(array_gambar[i - 1, j-1]) + self.index_check(array_gambar[i - 1, j]) + self.index_check(array_gambar[i - 1, j + 1]) \
                        + self.index_check(array_gambar[i, j - 1]) + -8*self.index_check(array_gambar[i, j]) + self.index_check(array_gambar[i, j + 1]) \
                        + self.index_check(array_gambar[i + 1, j - 1]) + self.index_check(array_gambar[i + 1, j]) + self.index_check(array_gambar[i + 1, j + 1])

                # temp = array_gambar[i-1, j-1]+array_gambar[i-1, j]+ array_gambar[i-1, j+1]+ \
                # array_gambar[i, j-1] +array_gambar[i, j]+ array_gambar[i, j+1]+ \
                # array_gambar[i+1, j-1] +array_gambar[i+1, j]+ array_gambar[i+1, j+1]

                for c in range(3):
                    if temp[c] < 0:
                        temp[c] = 0
                    elif temp[c] > 255:
                        temp[c] = 255
                    else:
                        temp[c] = temp[c]
                new_image[i, j] = temp

        new_image = Image.fromarray(new_image, "RGB")
        self.img = new_image
        self.show_image2()

    def normalin(self, x):
        if x < 0:
            x = 0
        elif x > 255:
            x = 255
        else:
            x = x
        return x


    def blur_it(self):
        array_gambar = np.array(self.img, dtype=np.float)

        new_image = np.zeros((array_gambar.shape[0], array_gambar.shape[1], array_gambar.shape[2]), dtype=np.uint8)

        for i in range(array_gambar.shape[0]-1):
            for j in range(array_gambar.shape[1]-1):
                rtemp = (self.index_check(array_gambar[i - 1, j - 1, 0]) + self.index_check(
                    array_gambar[i - 1, j, 0]) + self.index_check(array_gambar[i - 1, j + 1, 0])
                         + self.index_check(array_gambar[i, j - 1, 0]) + self.index_check(
                            array_gambar[i, j,0]) + self.index_check(array_gambar[i, j + 1, 0])
                         + self.index_check(array_gambar[i + 1, j - 1, 0]) + self.index_check(
                            array_gambar[i + 1, j,0]) + self.index_check(array_gambar[i + 1, j + 1, 0])) / 9

                gtemp = (self.index_check(array_gambar[i - 1, j - 1, 1]) + self.index_check(
                    array_gambar[i - 1, j, 1]) + self.index_check(array_gambar[i - 1, j + 1, 1])
                         + self.index_check(array_gambar[i, j - 1, 1]) + self.index_check(
                            array_gambar[i, j, 1]) + self.index_check(array_gambar[i, j + 1, 1])
                         + self.index_check(array_gambar[i + 1, j - 1, 1]) + self.index_check(
                            array_gambar[i + 1, j, 1]) + self.index_check(array_gambar[i + 1, j + 1, 1])) / 9

                btemp = (self.index_check(array_gambar[i - 1, j - 1, 2]) + self.index_check(
                    array_gambar[i - 1, j, 2]) + self.index_check(array_gambar[i - 1, j + 1, 2])
                         + self.index_check(array_gambar[i, j - 1, 2]) + self.index_check(
                            array_gambar[i, j, 2]) + self.index_check(array_gambar[i, j + 1, 2])
                         + self.index_check(array_gambar[i + 1, j - 1, 2]) + self.index_check(
                            array_gambar[i + 1, j, 2]) + self.index_check(array_gambar[i + 1, j + 1, 2])) / 9
            # temp = array_gambar[i-1, j-1]+array_gambar[i-1, j]+ array_gambar[i-1, j+1]+ \
            # array_gambar[i, j-1] +array_gambar[i, j]+ array_gambar[i, j+1]+ \
            # array_gambar[i+1, j-1] +array_gambar[i+1, j]+ array_gambar[i+1, j+1]
                self.normalin(rtemp)
                self.normalin(gtemp)
                self.normalin(btemp)

                new_image[i, j, 0] = rtemp
                new_image[i, j, 1] = gtemp
                new_image[i, j, 2] = btemp

        np.array(new_image, dtype=np.uint8)
        new_image = Image.fromarray(new_image, "RGB")
        self.img = new_image
        self.show_image2()

    def sharpen_it(self):
        array_gambar = np.array(self.img, dtype=np.float)
        new_image = np.zeros((array_gambar.shape[0], array_gambar.shape[1], array_gambar.shape[2]), dtype=np.int)

        for i in range(array_gambar.shape[0]-1):
            for j in range(array_gambar.shape[1]-1):
                rtemp = -self.index_check(array_gambar[i - 1, j, 0])\
                        - self.index_check(array_gambar[i, j - 1, 0])\
                        + 5*self.index_check(array_gambar[i, j,0]) \
                        - self.index_check(array_gambar[i, j + 1, 0])\
                        - self.index_check(array_gambar[i + 1, j,0])\

                gtemp = -self.index_check(array_gambar[i - 1, j, 1])\
                        - self.index_check(array_gambar[i, j - 1, 1])\
                        + 5*self.index_check(array_gambar[i, j,1]) \
                        - self.index_check(array_gambar[i, j + 1, 1])\
                        - self.index_check(array_gambar[i + 1, j,1])\

                btemp = -self.index_check(array_gambar[i - 1, j, 2])\
                        - self.index_check(array_gambar[i, j - 1, 2])\
                        + 5*self.index_check(array_gambar[i, j,2]) \
                        - self.index_check(array_gambar[i, j + 1, 2])\
                        - self.index_check(array_gambar[i + 1, j,2])\
                # gtemp = -1*self.index_check(array_gambar[i - 1, j, 1]) \
                #         + -1*self.index_check(array_gambar[i, j - 1, 1])\
                #         + 5*self.index_check(array_gambar[i, j, 1]) \
                #         + -1*self.index_check(array_gambar[i, j + 1, 1])\
                #         + -1*self.index_check(array_gambar[i + 1, j, 1]) \
                #
                # btemp = -1*self.index_check(array_gambar[i - 1, j, 2])\
                #         + -1*self.index_check(array_gambar[i, j - 1, 2])\
                #         + 5*self.index_check(array_gambar[i, j, 2])\
                #         + -1*self.index_check(array_gambar[i, j + 1, 2])\
                #         + -1*self.index_check(array_gambar[i + 1, j, 2]) \

                # rtemp = 0 * self.index_check(array_gambar[i - 1, j - 1, 0]) \
                #         + -1 * self.index_check(array_gambar[i - 1, j, 0]) \
                #         + 0 * self.index_check(array_gambar[i - 1, j + 1, 0]) \
                #         + -1 * self.index_check(array_gambar[i, j - 1, 0]) \
                #         + 5 * self.index_check(array_gambar[i, j, 0]) \
                #         + -1 * self.index_check(array_gambar[i, j + 1, 0]) \
                #         + 0 * self.index_check(array_gambar[i + 1, j - 1, 0]) \
                #         + -1 * self.index_check(array_gambar[i + 1, j, 0]) \
                #         + 0 * self.index_check(array_gambar[i + 1, j + 1, 0])
                #
                # gtemp = 0 * self.index_check(array_gambar[i - 1, j - 1, 1]) \
                #         + -1 * self.index_check(array_gambar[i - 1, j, 1]) \
                #         + 0 * self.index_check(array_gambar[i - 1, j + 1, 1]) \
                #         + -1 * self.index_check(array_gambar[i, j - 1, 1]) \
                #         + 5 * self.index_check(array_gambar[i, j, 1]) \
                #         + -1 * self.index_check(array_gambar[i, j + 1, 1]) \
                #         + 0 * self.index_check(array_gambar[i + 1, j - 1, 1]) \
                #         + -1 * self.index_check(array_gambar[i + 1, j, 1]) \
                #         + 0 * self.index_check(array_gambar[i + 1, j + 1, 1])
                #
                # btemp = 0 * self.index_check(array_gambar[i - 1, j - 1, 2]) \
                #         + -1 * self.index_check(array_gambar[i - 1, j, 2]) \
                #         + 0 * self.index_check(array_gambar[i - 1, j + 1, 2]) \
                #         + -1 * self.index_check(array_gambar[i, j - 1, 2]) \
                #         + 5 * self.index_check(array_gambar[i, j, 2]) \
                #         + -1 * self.index_check(array_gambar[i, j + 1, 2]) \
                #         + 0 * self.index_check(array_gambar[i + 1, j - 1, 2]) \
                #         + -1 * self.index_check(array_gambar[i + 1, j, 2]) \
                #         + 0 * self.index_check(array_gambar[i + 1, j + 1, 2])




                new_image[i, j, 0] = rtemp
                new_image[i, j, 1] = gtemp
                new_image[i, j, 2] = btemp

        new_image = np.clip(new_image,0,255)
        new_image = np.array(new_image, dtype=np.uint8)
        new_image = Image.fromarray(new_image, "RGB")
        self.img = new_image
        self.show_image2()

    def get_img_pixel(self,img, i, j, up, low):
        if img[i, j]>up:
            img[i, j] = 255
        # elif  img[i, j]>low and  img[i, j]<up:
        #     img[i, j] = 128
        elif img[i, j]<low:
            img[i, j] = 0

        return img[i, j]

    def segmentation_threshold(self):
        try:
            self.grayscale_img()
        finally:
            array_gambar = np.array(self.img, dtype=np.uint8)
            new_image = np.zeros((array_gambar.shape[0], array_gambar.shape[1]), dtype=np.int)
            upper_thres = self.entry_threshold_up.get()
            lower_thres = self.entry_threshold_low.get()

            for i in range(array_gambar.shape[0]):
                for j in range(array_gambar.shape[1]):
                    new_image[i,j] = self.get_img_pixel(array_gambar, i, j, int(upper_thres), int(lower_thres))


            new_image = np.clip(new_image,0,255)
            new_image = np.array(new_image, dtype=np.uint8)
            self.img = Image.fromarray(new_image, "L")
            self.show_image2()


    def set_gui(self):
        self.frame_open = Frame(self.master, bd=2)
        self.frame_open.grid(row=0, sticky=W)

        self.pathlabel = Label(self.frame_open, text='Image Path:')
        self.pathlabel.grid(row=0, column=0)

        self.entry_img = Entry(self.frame_open, width=80)
        self.entry_img.grid(row=0, column=1)

        Button(self.frame_open, text='Open', command=self.load_image).grid(row=0, column=2)
        Button(self.frame_open, text='Browse', command=self.browse_image).grid(row=0, column=3)
        Button(self.frame_open, text='Reset Images', command=self.load_image).grid(row=0, column=6)

        self.status = Label(self.frame_open, text="Image : None")
        self.status.grid(row=1, columnspan=4, sticky=W)


        self.histogram = Frame(self.master)
        self.histogram.grid(column=1, columnspan=3)

        Button(self.histogram, text='Histogram', command=self.histogram_it).pack(side=LEFT)
        Button(self.histogram, text='Histogram Equalization', command=self.histeq_it).pack()

        self.hcanvas1 = Canvas(self.histogram)
        self.hcanvas1.pack(side=RIGHT, fill=BOTH)

        self.fig = plt.figure(figsize=(4, 6), dpi=80)
        self.fig.tight_layout()
        grid = plt.GridSpec(3, 1, wspace=0.5, hspace=0.5)

        self.rhistogram = self.fig.add_subplot(grid[0,0])
        self.ghistogram = self.fig.add_subplot(grid[1,0])
        self.bhistogram = self.fig.add_subplot(grid[2,0])

        self.rhistogram.bar([0],[0])
        self.ghistogram.bar([0],[0])
        self.bhistogram.bar([0],[0])

        self.histogram_canvas = FigureCanvasTkAgg(self.fig, self.hcanvas1)
        self.histogram_canvas.draw()
        self.show_histogram()

        self.canvas_img = Canvas(self.master, height = self.gui_size[1], width=self.gui_size[0])
        self.canvas_img.grid(row=1, column=0)

        self.frame_tools = Frame(self.master, pady=3)
        self.frame_tools.grid(row=3, column=1)

        Button(self.frame_tools, text='Grayscale', command=self.greyscale_img).pack(side=LEFT, fill=Y)
        Button(self.frame_tools, text='Zoom In', command=self.zoom_in_img).pack(side=TOP, fill=X)
        Button(self.frame_tools, text='Zoom Out', command=self.zoom_out_img).pack(side=BOTTOM, fill=X)

        self.frame_inc_brightness = Frame(self.master, pady=3)
        self.frame_inc_brightness.grid(row=3, column=2)

        Button(self.frame_inc_brightness, text='Increase Brightness +', command=self.inc_brightness_plus).pack(side=TOP, fill=X)
        Button(self.frame_inc_brightness, text='Increase Brightness x', command=self.inc_brightness_times).pack(side=BOTTOM, fill=X)


        self.frame_dec_brightness = Frame(self.master, pady=3)
        self.frame_dec_brightness.grid(row=3, column=3)

        Button(self.frame_dec_brightness, text='Decrease Brightness -', command=self.dec_brightness_min).pack(side=TOP, fill=X)
        Button(self.frame_dec_brightness, text='Decrease Brightness /', command=self.dec_brightness_by).pack(side=BOTTOM, fill=X)

        self.frame_filter = Frame(self.master, pady=3)
        self.frame_filter.grid(row=3, column=6)

        Button(self.frame_filter, text='Edge Detection', command=self.edge_detect).pack(side=LEFT, fill=Y)
        Button(self.frame_filter, text='Blur', command=self.blur_it).pack(side=TOP, fill=X)
        Button(self.frame_filter, text='Sharpen', command=self.sharpen_it).pack(side=BOTTOM, fill=X)

        self.frame_seg_th = Frame(self.master, pady=3, bg="black")
        self.frame_seg_th.grid(row=1, column=6)

        Label(self.frame_seg_th, text="Segmentation", bg="black", fg="white").pack(fill=X)
        Button(self.frame_seg_th, text='Thresholding(Upper/Lower)', command=self.segmentation_threshold).pack(side=LEFT, fill=Y)
        self.entry_threshold_up = Entry(self.frame_seg_th, width=5)
        self.entry_threshold_up.pack(side=TOP, fill=X)
        self.entry_threshold_low = Entry(self.frame_seg_th, width=5)
        self.entry_threshold_low.pack(side=BOTTOM, fill=X)

        # self.frame_seg_ = Frame(self.master, pady=3)
        # self.frame_seg_.grid(row=2, column=6)
        #
        # Button(self.frame_seg_, text='Region Growth(X,Y)', command=self.sharpen_it).pack(side=LEFT, fill=X)
        # self.x_coordinate = Entry(self.frame_seg_, width=5)
        # self.x_coordinate.pack(side=TOP, fill=X)
        # self.y_coordinate = Entry(self.frame_seg_, width=5)
        # self.y_coordinate.pack(side=BOTTOM, fill=X)

        # self.frame_seg_2 = Frame(self.master,bg="Black")
        # self.frame_seg_2.grid(row=2, column=4)
        # Label(self.frame_seg_2, text="Threshold", bg="black", fg="white").pack(fill=X)
        # self.entry_threshold_up_reg = Entry(self.frame_seg_2, width=5)
        # self.entry_threshold_up_reg.pack(side=TOP, fill=X)
        # self.entry_threshold_low_reg = Entry(self.frame_seg_2, width=5)
        # self.entry_threshold_low_reg.pack(side=BOTTOM, fill=X)



        self.frame_move = Frame(self.master, pady=3, bg="Black")
        self.frame_move.grid(row=3, column=0, sticky=W)

        Label(self.frame_move,  text="Move Image", bg="black", fg="white").pack(fill=X)

        Button(self.frame_move, text='Up', bd=3, command=self.move_up).pack(fill=X)
        Button(self.frame_move, text='Left', bd=3, width=4, command=self.move_left).pack(side=LEFT)
        Button(self.frame_move, text='Right', bd=3, width=4, command=self.move_right).pack(side=RIGHT)
        Button(self.frame_move, text='Down', bd=3, width=4, command=self.move_down).pack(fill=X)



# run
window = Tk()
window.title("PCD Project - 1301150084")
application(window)
window.mainloop()